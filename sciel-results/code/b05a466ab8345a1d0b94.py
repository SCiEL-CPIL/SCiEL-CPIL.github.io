"""Rewritten program for 3x3x8 matrix multiplication tensor decomposition."""
import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p, rank):
    """Check that the decomposition exactly reproduces the multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build the tensor via einsum
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j]
                v_val = V_arr[j*p + k]
                w_val = W_arr[k*n + i]
                prod = np.dot(u_val, v_val)
                # Wait, that's not right. The tensor is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # So it's element-wise product summed over r
                s = 0.0
                for r in range(rank):
                    s += u_val[r] * v_val[r] * w_val[r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _validate_fast(U, V, W, n, m, p, rank):
    """Faster validation using numpy broadcasting."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is equivalent to: for each r, we have U_r (n*m), V_r (m*p), W_r (p*n)
    # The product tensor is sum_r outer(U_r reshaped to (n,m), V_r reshaped to (m,p), W_r reshaped to (p,n))
    
    # Let's do it via einsum
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _try_strassen_2x2x2(n, m, p):
    """Try to find a rank decomposition using Strassen-like 2x2x2 blocks.
    
    The idea: partition the 3x3x8 problem into sub-blocks and use Strassen's 7-term 
    formula for 2x2x2 matrix multiplication where applicable.
    
    For 3x3 * 3x8 -> 3x8:
    We can try to cover parts of the computation with Strassen blocks.
    
    A 2x2x2 Strassen block uses 7 multiplications instead of 8.
    If we can find non-overlapping 2x2x2 sub-blocks, we can save multiplications.
    
    Let's think about this more carefully.
    
    The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..2, j in 0..2, k in 0..7.
    
    We can partition rows of A into {0,1} and {2}, columns of B into various pairs, etc.
    
    Actually, let's try a systematic approach: try to embed multiple 2x2x2 Strassen blocks.
    """
    # Schoolbook rank is 3*3*8 = 72.
    # Each 2x2x2 Strassen block saves 1 multiplication (7 vs 8).
    # We need to find how many non-overlapping 2x2x2 blocks we can fit.
    
    # A 2x2x2 block covers: 2 rows of A, 2 columns of A (= 2 rows of B), 2 columns of B.
    # Specifically, block at (i1,i2, j1,j2, k1,k2) covers:
    #   A rows {i1,i2}, A cols {j1,j2} (which are B rows), B cols {k1,k2}
    #   Output: C rows {i1,i2}, C cols {k1,k2}
    
    # We have 3 rows of A, 3 cols of A, 8 cols of B.
    # Possible row pairs: (0,1), (0,2), (1,2)
    # Possible col-of-A pairs: (0,1), (0,2), (1,2)
    # Possible col-of-B pairs: C(8,2) = 28 pairs
    
    # Two blocks are non-overlapping if they don't share any (i,j,k) index in the tensor.
    # Block 1 covers indices: i in {i1,i2}, j in {j1,j2}, k in {k1,k2}
    # Block 2 covers indices: i in {i3,i4}, j in {j3,j4}, k in {k3,k4}
    # They overlap if ({i1,i2} & {i3,i4}) AND ({j1,j2} & {j3,j4}) AND ({k1,k2} & {k3,k4}) is non-empty.
    
    # For non-overlapping, we need at least one of the three sets to be disjoint.
    
    # Let's try to greedily find a set of non-overlapping blocks.
    
    row_pairs = [(0,1), (0,2), (1,2)]
    colA_pairs = [(0,1), (0,2), (1,2)]
    colB_pairs = [(k1, k2) for k1 in range(8) for k2 in range(k1+1, 8)]
    
    all_blocks = []
    for rp in row_pairs:
        for cp in colA_pairs:
            for kp in colB_pairs:
                all_blocks.append((rp, cp, kp))
    
    def blocks_overlap(b1, b2):
        r1, c1, k1 = b1
        r2, c2, k2 = b2
        # Overlap if all three dimensions share at least one index
        if set(r1) & set(r2) and set(c1) & set(c2) and set(k1) & set(k2):
            return True
        return False
    
    # Greedy: sort blocks and pick non-overlapping ones
    # Try multiple random orderings
    import random
    best_count = 0
    best_set = []
    
    for trial in range(100):
        random.shuffle(all_blocks)
        chosen = []
        for b in all_blocks:
            if all(not blocks_overlap(b, cb) for cb in chosen):
                chosen.append(b)
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_set = chosen[:]
    
    if best_count == 0:
        return None
    
    # Now construct the decomposition using best_set blocks + schoolbook for the rest
    # Start with schoolbook (rank 72), then for each Strassen block, replace 8 schoolbook terms with 7 Strassen terms.
    
    n, m, p = 3, 3, 8
    
    # Start with schoolbook
    U = {}
    V = {}
    W = {}
    # Each entry is a list of (value, column_index)
    
    # Schoolbook: for each (i,j,k), column c = i*24 + j*8 + k
    col_map = {}
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col_map[(i,j,k)] = c
                c += 1
    schoolbook_rank = c  # 72
    
    # For each Strassen block, we'll replace 8 schoolbook columns with 7 new columns.
    # Strassen 2x2x2:
    # Given A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # C = A*B
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # In terms of the tensor, for block at rows {i1,i2}, cols-of-A {j1,j2}, cols-of-B {k1,k2}:
    # The 8 schoolbook terms are (i1,j1,k1), (i1,j1,k2), (i1,j2,k1), (i1,j2,k2),
    # (i2,j1,k1), (i2,j1,k2), (i2,j2,k1), (i2,j2,k2)
    
    # We need to express these 8 products as 7 Strassen products.
    # The Strassen factors are:
    # m1: U part = (a11+a22) -> U[i1,j1] += 1, U[i2,j2] += 1; V part = (b11+b22) -> V[j1,k1] += 1, V[j2,k2] += 1; W part = 1
    # etc.
    
    # Actually, let me think about this differently. The Strassen decomposition for 2x2x2:
    # The 7 rank-1 tensors are:
    # 1. (a11+a22)(b11+b22) -> U: {(i1,j1):1, (i2,j2):1}, V: {(j1,k1):1, (j2,k2):1}, W: {(k1,i1):1, (k2,i2):1}
    # 2. (a21+a22)b11 -> U: {(i2,j1):1, (i2,j2):1}, V: {(j1,k1):1}, W: {(k1,i2):1}
    # 3. a11(b12-b22) -> U: {(i1,j1):1}, V: {(j1,k2):1, (j2,k2):-1}, W: {(k2,i1):1}
    # 4. a22(b21-b11) -> U: {(i2,j2):1}, V: {(j2,k1):1, (j1,k1):-1}, W: {(k1,i2):1}
    # 5. (a11+a12)b22 -> U: {(i1,j1):1, (i1,j2):1}, V: {(j2,k2):1}, W: {(k2,i1):1}
    # 6. (a21-a11)(b11+b12) -> U: {(i2,j1):1, (i1,j1):-1}, V: {(j1,k1):1, (j1,k2):1}, W: {(k1,i2):1, (k2,i2):1}
    # 7. (a12-a22)(b21+b22) -> U: {(i1,j2):1, (i2,j2):-1}, V: {(j2,k1):1, (j2,k2):1}, W: {(k1,i1):1, (k2,i1):1}
    
    # Let me verify: C11 = m1 + m4 - m5 + m7
    # m1 contributes to C11: W has (k1,i1):1, so yes.
    # m4 contributes to C11: W has (k1,i2):1, so no (we need i1). Wait, m4's W is (k1,i2).
    # Hmm, I think I made an error. Let me re-derive.
    
    # Actually, the standard Strassen formula gives:
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Where the indices are:
    # A = [[a11, a12], [a21, a22]] (rows 1,2; cols 1,2)
    # B = [[b11, b12], [b21, b22]] (rows 1,2; cols 1,2)
    # C = [[c11, c12], [c21, c22]]
    
    # In our tensor notation with 0-based indexing:
    # i1=0, i2=1, j1=0, j2=1, k1=0, k2=1 (local)
    
    # The 7 rank-1 terms:
    # t1: U: {(0,0):1, (1,1):1}, V: {(0,0):1, (1,1):1}, W: {(0,0):1, (1,1):1}
    # t2: U: {(1,0):1, (1,1):1}, V: {(0,0):1}, W: {(0,1):1}
    # t3: U: {(0,0):1}, V: {(0,1):1, (1,1):-1}, W: {(1,0):1}
    # t4: U: {(1,1):1}, V: {(1,0):1, (0,0):-1}, W: {(0,1):1}
    # t5: U: {(0,0):1, (0,1):1}, V: {(1,1):1}, W: {(1,0):1}
    # t6: U: {(1,0):1, (0,0):-1}, V: {(0,0):1, (0,1):1}, W: {(0,1):1, (1,1):1}
    # t7: U: {(0,1):1, (1,1):-1}, V: {(1,0):1, (1,1):1}, W: {(0,0):1, (1,0):1}
    
    # Let me verify c11 = t1 + t4 - t5 + t7:
    # t1 at (0,0): U(0,0)=1, V(0,0)=1, W(0,0)=1 -> product = 1
    # t4 at (0,0): U(0,0)=0, so 0
    # t5 at (0,0): U(0,0)=1, V(0,0)=0, so 0
    # t7 at (0,0): U(0,0)=0, so 0
    # Sum = 1. Correct (c11 should be a11*b11 + a12*b21, but this is just the (0,0) entry of the 2x2 product).
    
    # Wait, I'm confusing myself. The tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1 for the multiplication tensor.
    # For a 2x2x2 block, the 8 schoolbook terms each contribute 1 to their respective (i,j,k).
    # The Strassen decomposition should also give 1 for each of the 8 (i,j,k) indices.
    
    # Let me just trust the standard Strassen formula and implement it carefully.
    
    # For a block with local indices (i in {0,1}, j in {0,1}, k in {0,1}) mapped to global (i1+i, j1+j, k1+k):
    # The 7 Strassen terms (with local i,j,k):
    
    strassen_terms = [
        # (U dict, V dict, W dict) with local indices
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1, (1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,0):1}),
    ]
    
    # Build the decomposition
    # Start with all schoolbook terms, then for each Strassen block, remove the 8 schoolbook terms and add 7 Strassen terms.
    
    # Represent as: for each column, a list of (U_contrib, V_contrib, W_contrib)
    # Each contrib is a dict from index to value
    
    columns = []
    schoolbook_cols = {}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col_id = (i,j,k)
                U_d = {(i,j): 1}
                V_d = {(j,k): 1}
                W_d = {(k,i): 1}
                columns.append((U_d, V_d, W_d))
                schoolbook_cols[col_id] = len(columns) - 1
    
    # For each Strassen block, replace its 8 schoolbook columns with 7 Strassen columns
    for (rp, cp, kp) in best_set:
        i1, i2 = rp
        j1, j2 = cp
        k1, k2 = kp
        
        # Get the 8 schoolbook column indices
        block_cols = [
            schoolbook_cols[(i1,j1,k1)], schoolbook_cols[(i1,j1,k2)],
            schoolbook_cols[(i1,j2,k1)], schoolbook_cols[(i1,j2,k2)],
            schoolbook_cols[(i2,j1,k1)], schoolbook_cols[(i2,j1,k2)],
            schoolbook_cols[(i2,j2,k1)], schoolbook_cols[(i2,j2,k2)],
        ]
        
        # Mark them for removal (we'll rebuild)
        # Instead, let's build a new list of columns
        
    # This approach is getting complicated. Let me use a different strategy:
    # Build the full set of rank-1 terms (schoolbook + Strassen adjustments) and then deduplicate/combine.
    
    # Actually, the cleanest way: start with schoolbook, and for each Strassen block, 
    # we know the 8 schoolbook terms can be replaced by 7 Strassen terms.
    # So the net effect is: remove 8 columns, add 7 columns. Net reduction of 1 per block.
    
    # Let's just build it directly.
    
    all_terms = []  # Each term is (U_dict, V_dict, W_dict)
    
    # First, add all schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_terms.append(({(i,j):1}, {(j,k):1}, {(k,i):1}))
    
    # For each Strassen block, we need to "undo" the 8 schoolbook terms and "add" the 7 Strassen terms.
    # Since we're working with a multiset of rank-1 tensors, we can think of it as:
    # Final = Schoolbook - sum(block schoolbook terms) + sum(block Strassen terms)
    
    # But subtraction means adding the negative. Let's just build the final set.
    
    # Better approach: for each (i,j,k), determine if it's covered by a Strassen block.
    # If covered, don't add the schoolbook term; instead, the Strassen terms will cover it.
    
    covered = set()
    for (rp, cp, kp) in best_set:
        i1, i2 = rp
        j1, j2 = cp
        k1, k2 = kp
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i,j,k))
    
    # Add schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in covered:
                    all_terms.append(({(i,j):1}, {(j,k):1}, {(k,i):1}))
    
    # Add Strassen terms for each block
    for (rp, cp, kp) in best_set:
        i1, i2 = rp
        j1, j2 = cp
        k1, k2 = kp
        
        for (U_loc, V_loc, W_loc) in strassen_terms:
            U_g = {}
            for (li, lj), val in U_loc.items():
                gi, gj = i1 + li, j1 + lj
                U_g[(gi, gj)] = U_g.get((gi, gj), 0) + val
            V_g = {}
            for (lj, lk), val in V_loc.items():
                gj, gk = j1 + lj, k1 + lk
                V_g[(gj, gk)] = V_g.get((gj, gk), 0) + val
            W_g = {}
            for (lk, li), val in W_loc.items():
                gk, gi = k1 + lk, i1 + li
                W_g[(gk, gi)] = W_g.get((gk, gi), 0) + val
            all_terms.append((U_g, V_g, W_g))
    
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for r, (U_d, V_d, W_d) in enumerate(all_terms):
        for (i,j), val in U_d.items():
            U_mat[i*m + j][r] = val
        for (j,k), val in V_d.items():
            V_mat[j*p + k][r] = val
        for (k,i), val in W_d.items():
            W_mat[k*n + i][r] = val
    
    # Validate
    if _validate_fast(U_mat, V_mat, W_mat, n, m, p, rank):
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # First, try the Strassen-based approach
    result = _try_strassen_2x2x2(n, m, p)
    if result is not None:
        return result
    
    if time.time() > deadline:
        pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
