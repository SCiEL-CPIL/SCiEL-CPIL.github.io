"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_4x4():
    """
    Return a rank-49 decomposition for 4x4 matrix multiplication using
    Strassen's 2x2 algorithm applied to 2x2 blocks.
    Exact integer arithmetic.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen 7-term formulas for 2x2 block matmul
    # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    # For each of the 4 block positions (br,bc) of the 2x2 block grid,
    # and for each Strassen term (st), we create one rank-1 term.
    # But actually, the standard construction applies Strassen to the whole 4x4
    # by viewing it as 2x2 blocks of 2x2 matrices.
    
    # Let me use a cleaner formulation.
    # The 4x4 matmul: C = A*B where A is 4x4, B is 4x4, C is 4x4.
    # View A as 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    # View B as 2x2 blocks of 2x2: B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2.
    
    # Apply Strassen:
    # M1 = (A00+A11)(B00+B11)  -- 7 terms per block-pair, 4 block-pairs = 28? No.
    # Actually Strassen gives 7 products of 2x2 matrices, each requiring 4 scalar mults.
    # Total = 7*4 = 28? No, that's not right either.
    
    # Let me be precise. Each "product" in Strassen is a 2x2 matrix product.
    # A 2x2 matrix product can be done in rank 7 (Strassen) or rank 4 (schoolbook).
    # If we use schoolbook for the inner 2x2 products: 7 * 4 = 28.
    # If we use Strassen for the inner 2x2 products: 7 * 7 = 49.
    
    # The rank-49 construction uses Strassen at BOTH levels.
    # The rank-28 construction uses Strassen at the outer level, schoolbook at inner.
    
    # Wait, that's not right. Let me reconsider.
    # 4x4 matmul as 2x2 blocks of 2x2:
    # Outer Strassen: 7 products of 2x2 matrices
    # Each 2x2 product: if done by schoolbook, rank 4
    # Total rank: 7 * 4 = 28? That can't be right since rank(4x4) >= 47.
    
    # I think I'm confusing myself. Let me use the standard approach.
    
    # The tensor for 4x4 matmul has dimension 16 x 16 x 16.
    # The rank-49 construction from Strassen:
    # Decompose into 4 independent 2x2 block matmuls? No.
    
    # Actually, the correct way: 4x4 matmul can be seen as computing 16 output entries.
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # 
    # Strassen's algorithm for 4x4:
    # Partition into 2x2 blocks. Apply Strassen's 7 multiplications to the blocks.
    # Each block multiplication is a 2x2 matrix product.
    # A 2x2 matrix product has tensor rank 7 (Strassen) or 4 (schoolbook).
    # 
    # If we use Strassen for the block products: 7 (outer) * 7 (inner) = 49.
    # If we use schoolbook for the block products: 7 (outer) * 4 (inner) = 28.
    #
    # But rank(4x4) is known to be at least 47, so 28 is impossible.
    # The issue: the outer Strassen combines blocks in a way that the inner products
    # are not independent 2x2 matmuls in the tensor sense.
    
    # Let me just directly construct the rank-49 decomposition.
    
    # I'll use the formulation where we have 49 rank-1 terms.
    # Each term corresponds to a choice of (block_row, block_col, strassen_inner_term).
    # 
    # Actually, let me use a more direct approach. The 4x4 matmul tensor
    # T[i,j,k] = delta_{i,k'} * delta_{k',j} ... no.
    
    # T[a,b,c] where a = i*m+j (row i, col j of A), b = j'*p+k (row j', col k of B),
    # c = k'*n+i' (row k', col i' of C).
    # T[a,b,c] = 1 if j==j' and k==k' and i==i', else 0.
    
    # For the rank-49 Strassen construction, I'll build it systematically.
    
    # Let me use the approach from the original code which worked.
    
    def make_block_A(br, bc, sign=1):
        """2x2 block of A at block position (br,bc), as dict of flat_index->coeff"""
        result = {}
        for i in range(2):
            for j in range(2):
                row = 2 * br + i
                col_idx = 2 * bc + j
                idx = row * m + col_idx
                result[idx] = result.get(idx, 0) + sign
        return result
    
    def make_block_B(br, bc, sign=1):
        """2x2 block of B at block position (br,bc), as dict of flat_index->coeff"""
        result = {}
        for i in range(2):
            for j in range(2):
                row = 2 * br + i
                col_idx = 2 * bc + j
                idx = row * p + col_idx
                result[idx] = result.get(idx, 0) + sign
        return result
    
    def combine(f1, f2, sign2=1):
        result = dict(f1)
        for idx, c in f2.items():
            result[idx] = result.get(idx, 0) + c * sign2
        return result
    
    def subtract(f1, f2):
        return combine(f1, f2, -1)
    
    # 7 Strassen X factors (combinations of A blocks)
    # X1 = A00+A11, X2 = A10+A11, X3 = A00, X4 = A11,
    # X5 = A00+A01, X6 = A10-A00, X7 = A01-A11
    X_factors = [
        combine(make_block_A(0,0), make_block_A(1,1)),   # X1
        combine(make_block_A(1,0), make_block_A(1,1)),   # X2
        make_block_A(0,0),                                # X3
        make_block_A(1,1),                                # X4
        combine(make_block_A(0,0), make_block_A(0,1)),    # X5
        subtract(make_block_A(1,0), make_block_A(0,0)),   # X6
        subtract(make_block_A(0,1), make_block_A(1,1)),   # X7
    ]
    
    # 7 Strassen Y factors (combinations of B blocks)
    # Y1 = B00+B11, Y2 = B00, Y3 = B01-B11, Y4 = B10-B00,
    # Y5 = B11, Y6 = B00+B01, Y7 = B10+B11
    Y_factors = [
        combine(make_block_B(0,0), make_block_B(1,1)),   # Y1
        make_block_B(0,0),                                # Y2
        subtract(make_block_B(0,1), make_block_B(1,1)),   # Y3
        subtract(make_block_B(1,0), make_block_B(0,0)),   # Y4
        make_block_B(1,1),                                # Y5
        combine(make_block_B(0,0), make_block_B(0,1)),    # Y6
        combine(make_block_B(1,0), make_block_B(1,1)),    # Y7
    ]
    
    # For each Strassen term (s=0..6), the 2x2 block products X_s * Y_s
    # contribute to output blocks. Each 2x2 block product (X_s * Y_s) is
    # itself a 2x2 matrix product, which we decompose using Strassen (rank 7)
    # or schoolbook (rank 4).
    
    # For rank-49: use Strassen (rank 7) for each of the 7 block products.
    # Total: 7 * 7 = 49 terms.
    
    # For each block product X_s * Y_s (where X_s, Y_s are 2x2 matrices),
    # apply Strassen again:
    # Let X_s = [[x00,x01],[x10,x11]], Y_s = [[y00,y01],[y10,y11]]
    # N1 = (x00+x11)(y00+y11)
    # N2 = (x10+x11)y00
    # N3 = x00(y01-y11)
    # N4 = x11(y10-y00)
    # N5 = (x00+x01)y11
    # N6 = (x10-x00)(y00+y01)
    # N7 = (x01-x11)(y10+y11)
    # Z00 = N1+N4-N5+N7, Z01 = N3+N5, Z10 = N2+N4, Z11 = N1-N2+N3+N6
    
    # The 2x2 output block (X_s*Y_s) contributes to specific C blocks.
    # C00 gets contributions from: s=0 (coeff +1), s=3 (coeff +1), s=4 (coeff -1), s=6 (coeff +1)
    # C01 gets contributions from: s=2 (coeff +1), s=4 (coeff +1)
    # C10 gets contributions from: s=1 (coeff +1), s=3 (coeff +1)
    # C11 gets contributions from: s=0 (coeff +1), s=1 (coeff -1), s=2 (coeff +1), s=5 (coeff +1)
    
    block_contrib = [
        {0: 1, 3: 1, 4: -1, 6: 1},   # contributes to C00
        {1: 1, 2: 1},                 # contributes to C01
        {1: 1, 3: 1},                 # contributes to C10
        {0: 1, 1: -1, 2: 1, 5: 1},   # contributes to C11
    ]
    
    # Strassen inner coefficients for the 2x2 product X_s*Y_s:
    # N1 -> Z00(+1), Z11(+1)
    # N2 -> Z10(+1), Z11(-1)
    # N3 -> Z01(+1), Z11(+1)
    # N4 -> Z00(+1), Z10(+1)
    # N5 -> Z00(-1), Z01(+1)
    # N6 -> Z11(+1)
    # N7 -> Z00(+1)
    
    inner_strassen = [
        {0: 1, 3: 1},    # N1: Z00, Z11
        {2: 1, 3: -1},   # N2: Z10, Z11
        {1: 1, 3: 1},    # N3: Z01, Z11
        {0: 1, 2: 1},    # N4: Z00, Z10
        {0: -1, 1: 1},   # N5: Z00, Z01
        {3: 1},          # N6: Z11
        {0: 1},          # N7: Z00
    ]
    
    # Inner Strassen X factors for 2x2 product:
    # nX1 = x00+x11, nX2 = x10+x11, nX3 = x00, nX4 = x11,
    # nX5 = x00+x01, nX6 = x10-x00, nX7 = x01-x11
    # Inner Strassen Y factors:
    # nY1 = y00+y11, nY2 = y00, nY3 = y01-y11, nY4 = y10-y00,
    # nY5 = y11, nY6 = y00+y01, nY7 = y10+y11
    
    # For each outer Strassen term s (0..6) and inner Strassen term t (0..6):
    # The U factor is: (inner X factor applied to X_s's 2x2 entries)
    # The V factor is: (inner Y factor applied to Y_s's 2x2 entries)
    # The W factor is: (outer block contribution) * (inner strassen coefficient)
    
    # X_s's 2x2 entries as dicts:
    # X_s is a 2x2 matrix where each entry is a linear form in A.
    # X_s[i][j] = sum of A entries with coefficients.
    
    def get_X_s_entries(s):
        """Get the 4 entries of X_s as dicts (linear forms in A)"""
        # X_factors[s] is a flat dict. We need to split into 2x2.
        # X_s[i][j] corresponds to A[2*br+i, 2*bc+j] for some (br,bc)
        # But X_s is a combination of blocks, so each entry is a sum.
        
        # Actually, X_factors[s] is already a dict mapping flat A index -> coeff.
        # We need to express the 2x2 entries of X_s.
        # X_s is a 2x2 matrix. Its (i,j) entry is a linear form in A.
        # 
        # For the inner Strassen, we need x00, x01, x10, x11 of X_s.
        # These are 2x2 blocks of the original A, combined according to Strassen.
        
        # Let me restructure. X_s is a 2x2 matrix of linear forms.
        # X_s[i][j] = linear form in A.
        
        # For s=0: X0 = A00+A11 (2x2 matrix sum)
        #   X0[0][0] = A[0][0]+A[2][2], X0[0][1] = A[0][1]+A[2][3]
        #   X0[1][0] = A[1][0]+A[3][2], X0[1][1] = A[1][1]+A[3][3]
        
        # I'll build these explicitly.
        blocks = {
            0: make_block_A(0, 0),  # A00
            1: make_block_A(0, 1),  # A01
            2: make_block_A(1, 0),  # A10
            3: make_block_A(1, 1),  # A11
        }
        
        # Outer Strassen X factors as combinations of blocks
        X_combos = [
            (0, 1, 3, 1),   # X0 = A00 + A11
            (2, 1, 3, 1),   # X1 = A10 + A11
            (0, 1),         # X2 = A00
            (3, 1),         # X3 = A11
            (0, 1, 1, 1),   # X4 = A00 + A01
            (2, 1, 0, -1),  # X5 = A10 - A00
            (1, 1, 3, -1),  # X6 = A01 - A11
        ]
        
        # Build 2x2 entries
        entries = [[{} for _ in range(2)] for _ in range(2)]
        combo = X_combos[s]
        for br in range(2):
            for bc in range(2):
                block_id = br * 2 + bc
                if block_id == 0:
                    coeff = 1 if (0 in combo) else 0
                elif block_id == 1:
                    coeff = 1 if (1 in combo) else 0
                elif block_id == 2:
                    coeff = 1 if (2 in combo) else 0
                elif block_id == 3:
                    coeff = 1 if (3 in combo) else 0
                else:
                    coeff = 0
                
                # Handle signs properly
                # This is getting complicated. Let me use a different approach.
        
        # Simpler: just use the flat dict and figure out the 2x2 structure.
        # X_factors[s] is a dict {flat_A_index: coeff}
        # flat_A_index = row * 4 + col
        # The 2x2 entries of X_s correspond to specific (row, col) ranges.
        
        x_dict = X_factors[s]
        x_entries = [[{} for _ in range(2)] for _ in range(2)]
        for idx, coeff in x_dict.items():
            row, col = divmod(idx, m)
            br, bc = row // 2, col // 2
            ir, ic = row % 2, col % 2
            x_entries[ir][ic][idx] = x_entries[ir][ic].get(idx, 0) + coeff
        
        return x_entries
    
    def get_Y_s_entries(s):
        """Get the 4 entries of Y_s as dicts (linear forms in B)"""
        y_dict = Y_factors[s]
        y_entries = [[{} for _ in range(2)] for _ in range(2)]
        for idx, coeff in y_dict.items():
            row, col = divmod(idx, p)
            br, bc = row // 2, col // 2
            ir, ic = row % 2, col % 2
            y_entries[ir][ic][idx] = y_entries[ir][ic].get(idx, 0) + coeff
        return y_entries
    
    # Precompute entries for all s
    X_entries = [get_X_s_entries(s) for s in range(7)]
    Y_entries = [get_Y_s_entries(s) for s in range(7)]
    
    # Inner Strassen X factor combinations (as (x00, x01, x10, x11) operations)
    # nX1 = x00+x11, nX2 = x10+x11, nX3 = x00, nX4 = x11,
    # nX5 = x00+x01, nX6 = x10-x00, nX7 = x01-x11
    def inner_X_factor(x_entries, t):
        """Get the inner Strassen X factor as a flat dict"""
        x00, x01 = x_entries[0]
        x10, x11 = x_entries[1]
        if t == 0:
            return combine(x00, x11)
        elif t == 1:
            return combine(x10, x11)
        elif t == 2:
            return x00
        elif t == 3:
            return x11
        elif t == 4:
            return combine(x00, x01)
        elif t == 5:
            return subtract(x10, x00)
        elif t == 6:
            return subtract(x01, x11)
    
    def inner_Y_factor(y_entries, t):
        """Get the inner Strassen Y factor as a flat dict"""
        y00, y01 = y_entries[0]
        y10, y11 = y_entries[1]
        if t == 0:
            return combine(y00, y11)
        elif t == 1:
            return y00
        elif t == 2:
            return subtract(y01, y11)
        elif t == 3:
            return subtract(y10, y00)
        elif t == 4:
            return y11
        elif t == 5:
            return combine(y00, y01)
        elif t == 6:
            return combine(y10, y11)
    
    # Build the decomposition
    for s in range(7):
        for t in range(7):
            # U factor
            u_factor = inner_X_factor(X_entries[s], t)
            for idx, coeff in u_factor.items():
                if coeff != 0:
                    U[idx][col] += coeff
            
            # V factor
            v_factor = inner_Y_factor(Y_entries[s], t)
            for idx, coeff in v_factor.items():
                if coeff != 0:
                    V[idx][col] += coeff
            
            # W factor
            # The outer Strassen term s contributes to certain C blocks with certain coefficients.
            # The inner Strassen term t contributes to certain 2x2 positions within those blocks.
            
            # For each C block that s contributes to:
            for c_block, outer_coeff in block_contrib[s].items():
                # c_block: 0=C00, 1=C01, 2=C10, 3=C11
                # Inner Strassen term t contributes to positions within the 2x2 block
                for pos, inner_coeff in inner_strassen[t].items():
                    # pos: 0=Z00, 1=Z01, 2=Z10, 3=Z11 (within the 2x2 block)
                    total_coeff = outer_coeff * inner_coeff
                    if total_coeff != 0:
                        # Map (c_block, pos) to flat C index
                        # c_block 0 -> rows 0-1, cols 0-1
                        # c_block 1 -> rows 0-1, cols 2-3
                        # c_block 2 -> rows 2-3, cols 0-1
                        # c_block 3 -> rows 2-3, cols 2-3
                        c_br = c_block // 2
                        c_bc = c_block % 2
                        pos_r = pos // 2
                        pos_c = pos % 2
                        C_row = 2 * c_br + pos_r
                        C_col = 2 * c_bc + pos_c
                        W_idx = C_col * n + C_row
                        W[W_idx][col] += total_coeff
            
            col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    return U, V, W


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using structured checks."""
    import numpy as np
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check all nonzero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(np.dot(U_np[a], V_np[b]), W_np[c])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check a subset of zero entries
    import random
    rng = random.Random(12345)
    for _ in range(500):
        a = rng.randint(0, n*m - 1)
        b = rng.randint(0, m*p - 1)
        c = rng.randint(0, p*n - 1)
        i, j = divmod(a, m)
        j2, k = divmod(b, p)
        k2, i2 = divmod(c, n)
        if not (j == j2 and k == k2 and i == i2):
            val = np.dot(np.dot(U_np[a], V_np[b]), W_np[c])
            if abs(val) > tol:
                return False
    
    return True


def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification - check all entries."""
    import numpy as np
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build full tensor
    T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np, optimize=True)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    
    # Check all zero entries
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                if not (j == j2 and k == k2 and i == i2):
                    if abs(T[a, b, c]) > tol:
                        return False
    
    return True


def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition
    for 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try the rank-49 Strassen-based decomposition
    try:
        U, V, W = _strassen_4x4()
        
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        
        # Fast verification first
        if _verify_fast(U, V, W, n, m, p):
            # Do full verification if time permits
            if time.time() < deadline - 5:
                if _verify_full(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
            else:
                return {"U": U, "V": V, "W": W}
        else:
            # Fast check failed, try full
            if _verify_full(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 64
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
