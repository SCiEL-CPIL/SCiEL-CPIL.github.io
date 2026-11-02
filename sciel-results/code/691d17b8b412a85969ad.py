"""Matrix multiplication 2x3x10 - rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    
    r = len(U[0]) if U else 0
    if r == 0:
        return False
        
    # Convert to numpy for faster verification if large, but loop is fine for small
    # U shape: (n*m, r)
    # V shape: (m*p, r)
    # W shape: (p*n, r)
    
    # Check target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for col in range(r):
                    total += U[a][col] * V[b][col] * W[c][col]
                if abs(total - 1) > 1e-9:
                    return False
    
    # Check non-target entries are 0
    # We can optimize this check by constructing the full tensor T from U,V,W
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # We need T[a,b,c] == 1 if c == k*n+i for a=i*m+j, b=j*p+k
    # else 0
    
    # For small sizes, direct loop is safe.
    # Let's do a quick numpy check for speed if r is not huge
    if r > 10:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Reshape to (n, m, r), (m, p, r), (p, n, r)
        U_reshaped = U_np.reshape(n, m, r)
        V_reshaped = V_np.reshape(m, p, r)
        W_reshaped = W_np.reshape(p, n, r)
        
        # Compute tensor T
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # This is a contraction.
        # Use einsum
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return False
            
        # Check against target
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Target: T[i,j,k] should be 1
                    if abs(T[i,j,k] - 1) > 1e-9:
                        return False
                    # Non-targets: For fixed i,j, other k' != k should be 0
                    # For fixed i,k, other j' != j should be 0
                    # For fixed j,k, other i' != i should be 0
                    # The einsum result T[i,j,k] corresponds to output entry C[i,k] from input A[i,j] and B[j,k].
                    # The condition is:
                    # sum_j A[i,j] B[j,k] = C[i,k]
                    # The tensor T[i,j,k] is the coefficient of A[i,j]B[j,k] in C[i,k].
                    # So T[i,j,k] must be 1.
                    # All other combinations (i', j', k') not corresponding to valid matrix mult terms must be 0?
                    # No, the tensor is defined on indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
                    # The non-zero entries are exactly those (i, j, k).
                    # Wait, the tensor space is U_n x V_m x W_p.
                    # The basis vectors are e_i, f_j, g_k.
                    # The matmul tensor T = sum_{i,j,k} e_i \otimes f_j \otimes g_{k*n+i}? 
                    # No, the output index is c = k*n + i.
                    # The tensor is T = sum_{i,j,k} e_i \otimes f_j \otimes h_{k*n+i}
                    # where h_c are basis vectors for the output space of dimension p*n.
                    
                    # In my verification loop above:
                    # a = i*m+j
                    # b = j*p+k
                    # c = k*n+i
                    # This maps (i,j,k) to indices in the flattened spaces.
                    # The "non-target" entries in the flattened tensor space are any (a,b,c) 
                    # that do NOT correspond to a valid (i,j,k) triple.
                    
                    # My numpy check T[i,j,k] only checks the "valid" triples.
                    # I need to ensure that for any (a,b,c) where c != k*n+i (given i,j,k derived from a,b),
                    # the sum is 0.
                    
                    # Actually, the einsum 'ijr,jkr,kir->ijk' computes exactly the coefficients for the valid (i,j,k).
                    # It does NOT compute the invalid ones.
                    # To check invalid ones, I need to verify that the decomposition respects the structure.
                    # Let's stick to the loop for correctness, it's fast enough for r~55.
                    pass
    
    # Fallback to loop for full correctness check including zeros
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                for c_test in range(p * n):
                    # Determine if (i,j,k_test) is a valid target for this a,b
                    # a = i*m+j is fixed. b = j*p+k is fixed.
                    # The only valid c for this (a,b) is c_valid = k*n + i.
                    if c_test == k * n + i:
                        continue
                    total = 0
                    for col in range(r):
                        total += U[a][col] * V[b][col] * W[c_test][col]
                    if abs(total) > 1e-9:
                        return False
    return True

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Generate schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # Strategy: Decompose the 2x3 @ 3x10 multiplication into smaller blocks.
    # Let A be 2x3, B be 3x10.
    # Split B into B1 (3x5) and B2 (3x5).
    # C = A @ B1 + A @ B2.
    # This doesn't help directly.
    
    # Better split: Split the 3x10 matrix B into two 3x2 matrices and one 3x4?
    # Or split the 2x3 matrix A?
    
    # Let's look at the rank of small rectangular multiplications.
    # R(2x2 @ 2x2) = 7 (Strassen).
    # R(2x1 @ 1x2) = 2 (Schoolbook, minimal).
    # R(1x1 @ 1x1) = 1.
    
    # Can we decompose 2x3 @ 3x10 into blocks where we use Strassen?
    # The dimensions are 2, 3, 10.
    # 3 is not a power of 2. 10 is not a power of 2.
    
    # Consider splitting the middle dimension m=3 into 2+1.
    # A = [A_left (2x2), A_right (2x1)]
    # B = [B_top (2x10); B_bottom (1x10)]
    # C = A_left @ B_top + A_right @ B_bottom
    
    # Term 1: 2x2 @ 2x10
    # Term 2: 2x1 @ 1x10
    
    # Rank(Term 2):
    # 2x1 @ 1x10 is just 20 scalar multiplications (each a_i * b_j).
    # R(2x1 @ 1x10) = 20.
    
    # Rank(Term 1):
    # 2x2 @ 2x10.
    # This is equivalent to 10 independent 2x2 @ 2x1 multiplications?
    # No, B_top is 2x10.
    # Let's split B_top into 5 blocks of 2x2.
    # B_top = [B_t1, B_t2, B_t3, B_t4, B_t5], each 2x2.
    # C_part = A_left @ B_t1 + A_left @ B_t2 + ...
    # Since A_left is fixed, can we share computations?
    # A_left is 2x2.
    # For each 2x2 block B_ti, we compute A_left @ B_ti.
    # This is a 2x2 @ 2x2 multiplication.
    # Using Strassen, each 2x2 @ 2x2 costs 7 multiplications.
    # There are 5 such blocks.
    # Total cost for Term 1 = 5 * 7 = 35.
    
    # Total Rank = 35 (Term 1) + 20 (Term 2) = 55.
    
    # This matches the "rank-55" inspiration!
    # Let's construct this decomposition explicitly.
    
    U_new = []
    V_new = []
    W_new = []
    
    # We will build the decomposition column by column in the rank space.
    # Each "multiplication" in the algorithm corresponds to one column in U, V, W.
    
    # Term 2: A_right (2x1) @ B_bottom (1x10)
    # A_right = [a02, a12]^T
    # B_bottom = [b02, b12, ..., b92] (row vector)
    # C0 += a02 * b02 + ... + a02 * b92
    # C1 += a12 * b02 + ... + a12 * b92
    
    # For each k in 0..9:
    # Multiplication m_k = a02 * b_k2
    # This contributes to C0.
    # Multiplication n_k = a12 * b_k2
    # This contributes to C1.
    
    # So for each k:
    # Col k: U[0, col] = 1 (a02), V[k*1+0, col] = 1 (b_k2), W[0*2+0, col] = 1 (C0)
    # Wait, indices:
    # a = i*m+j. For A_right, i=0, j=2 -> a=2. i=1, j=2 -> a=5.
    # b = j*p+k. For B_bottom, j=2. b = 2*10+k = 20+k.
    # c = k*n+i.
    
    # Let's add 20 columns for Term 2.
    for k in range(10):
        # a02 * b_k2 -> C0
        U_new.append([0]*60) # Placeholder, will resize
        V_new.append([0]*60)
        W_new.append([0]*60)
        
        # We are building lists of rows.
        # Let's just append to lists and pad later, or build matrices directly.
        # It's easier to build lists of vectors.
        
    # Let's restart the construction properly.
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    # --- Term 2: A_right (2x1) @ B_bottom (1x10) ---
    # A_right indices in A (2x3): (0,2) -> idx 2, (1,2) -> idx 5
    # B_bottom indices in B (3x10): (2,k) -> idx 20+k
    # Output indices in C (2x10): (0,k) -> idx k*2+0, (1,k) -> idx k*2+1
    
    for k in range(10):
        # Mult 1: a_02 * b_2k -> c_0k
        col_U = [0] * (n*m)
        col_V = [0] * (m*p)
        col_W = [0] * (p*n)
        
        col_U[2] = 1
        col_V[20+k] = 1
        col_W[k*2+0] = 1
        
        cols_U.append(col_U)
        cols_V.append(col_V)
        cols_W.append(col_W)
        
        # Mult 2: a_12 * b_2k -> c_1k
        col_U[5] = 1
        # col_V is same
        col_W[k*2+1] = 1
        
        cols_U.append(col_U)
        cols_V.append(col_V)
        cols_W.append(col_W)
        
    # Current rank: 20
    
    # --- Term 1: A_left (2x2) @ B_top (2x10) ---
    # A_left indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    # B_top indices: (0,k)->k, (1,k)->10+k
    # Output indices: (0,k)->k*2+0, (1,k)->k*2+1
    
    # Split B_top into 5 blocks of width 2.
    # Block q covers columns k=2q, 2q+1.
    for q in range(5):
        k0 = 2*q
        k1 = 2*q + 1
        
        # A_left = [[a00, a01], [a10, a11]]
        # B_block = [[b0k0, b0k1], [b1k0, b1k1]]
        
        # Strassen for 2x2 @ 2x2:
        # m1 = (a00 + a11) * (b00 + b10)  -- Wait, standard Strassen is for C = A*B
        # Let's use the standard 7-multiplication formula.
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-h)
        # m5 = (a+b)h
        # m6 = (c-d)(e+f)
        # m7 = (b-d)(g+h)
        
        # C11 = m1 - m2 + m4 + m7
        # C12 = m3 + m5
        # C21 = m2 + m6
        # C22 = m1 - m5 + m3 - m6
        
        # Here:
        # a = a00, b = a01, c = a10, d = a11
        # e = b0k0, f = b0k1, g = b1k0, h = b1k1
        
        # We need to add 7 columns for each block.
        
        # Helper to add a column
        def add_col(u_vals, v_vals, w_vals):
            col_U = [0] * (n*m)
            col_V = [0] * (m*p)
            col_W = [0] * (p*n)
            for idx, val in u_vals.items():
                col_U[idx] += val
            for idx, val in v_vals.items():
                col_V[idx] += val
            for idx, val in w_vals.items():
                col_W[idx] += val
            cols_U.append(col_U)
            cols_V.append(col_V)
            cols_W.append(col_W)
            
        # Indices:
        # A: a00=0, a01=1, a10=3, a11=4
        # B: b0k0=k0, b0k1=k1, b1k0=10+k0, b1k1=10+k1
        # C: c0k0=k0*2, c0k1=k1*2, c1k0=k0*2+1, c1k1=k1*2+1
        
        # m1 = (a00 + a11) * (b0k0 + b1k0)
        # U: {0:1, 4:1}
        # V: {k0:1, 10+k0:1}
        # W: contributes to C11 (m1), C22 (m1)
        # C11 index: k0*2+0? No, C[i,k]. i=0, k=k0 -> idx k0*2.
        # C22 index: i=1, k=k0 -> idx k0*2+1.
        # Wait, Strassen outputs are 2x2.
        # C_block = [[C0k0, C0k1], [C1k0, C1k1]]
        # m1 contributes +1 to C0k0 and +1 to C1k1.
        
        add_col({0:1, 4:1}, {k0:1, 10+k0:1}, {k0*2:1, k1*2+1:1})
        
        # m2 = (a10 + a11) * b0k0
        # U: {3:1, 4:1}
        # V: {k0:1}
        # W: contributes -1 to C0k0, +1 to C1k0
        add_col({3:1, 4:1}, {k0:1}, {k0*2:-1, k0*2+1:1})
        
        # m3 = a00 * (b0k1 - b1k1)
        # U: {0:1}
        # V: {k1:1, 10+k1:-1}
        # W: contributes +1 to C0k1, +1 to C1k1
        add_col({0:1}, {k1:1, 10+k1:-1}, {k1*2:1, k1*2+1:1})
        
        # m4 = a11 * (b1k0 - b1k1)
        # U: {4:1}
        # V: {10+k0:1, 10+k1:-1}
        # W: contributes +1 to C0k0, -1 to C1k1
        add_col({4:1}, {10+k0:1, 10+k1:-1}, {k0*2:1, k1*2+1:-1})
        
        # m5 = (a00 + a01) * b1k1
        # U: {0:1, 1:1}
        # V: {10+k1:1}
        # W: contributes +1 to C0k1, -1 to C1k1
        add_col({0:1, 1:1}, {10+k1:1}, {k1*2:1, k1*2+1:-1})
        
        # m6 = (a10 - a11) * (b0k0 + b0k1)
        # U: {3:1, 4:-1}
        # V: {k0:1, k1:1}
        # W: contributes +1 to C1k0, -1 to C1k1
        add_col({3:1, 4:-1}, {k0:1, k1:1}, {k0*2+1:1, k1*2+1:-1})
        
        # m7 = (a01 - a11) * (b1k0 + b1k1)
        # U: {1:1, 4:-1}
        # V: {10+k0:1, 10+k1:1}
        # W: contributes +1 to C0k0, -1 to C1k1
        add_col({1:1, 4:-1}, {10+k0:1, 10+k1:1}, {k0*2:1, k1*2+1:-1})
        
    # Total rank = 20 + 5*7 = 55.
    
    U_new = [cols_U[i] for i in range(len(cols_U))]
    V_new = [cols_V[i] for i in range(len(cols_V))]
    W_new = [cols_W[i] for i in range(len(cols_W))]
    
    # Transpose to (rows, rank) format expected by evaluator
    # Current cols_U is list of rows? No, cols_U is list of columns (vectors of length n*m).
    # We need U to be (n*m, rank).
    # So U[row][col] = cols_U[col][row]
    
    rank_found = len(cols_U)
    U_mat = [[cols_U[c][r] for c in range(rank_found)] for r in range(n*m)]
    V_mat = [[cols_V[c][r] for c in range(rank_found)] for r in range(m*p)]
    W_mat = [[cols_W[c][r] for c in range(rank_found)] for r in range(p*n)]
    
    if _verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    # If verification fails, return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
