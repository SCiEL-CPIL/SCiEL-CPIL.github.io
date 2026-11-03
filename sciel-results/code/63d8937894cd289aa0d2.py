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
        
    # Convert to numpy for faster verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to (n, m, r), (m, p, r), (p, n, r)
    U_reshaped = U_np.reshape(n, m, r)
    V_reshaped = V_np.reshape(m, p, r)
    W_reshaped = W_np.reshape(p, n, r)
    
    # Compute tensor T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check target entries: T[i, j, k] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, j, k] - 1) > 1e-9:
                    return False
    
    # Check non-target entries are 0
    # The tensor space is U_n x V_m x W_p.
    # The basis vectors are e_i, f_j, g_k.
    # The matmul tensor T = sum_{i,j,k} e_i \otimes f_j \otimes g_{k*n+i}
    # In the flattened index space:
    # a = i*m+j, b = j*p+k, c = k*n+i
    # We need T_flat[a, b, c] = 1 if c == k*n+i for the given i,j,k derived from a,b
    # and 0 otherwise.
    
    # Let's construct the full flattened tensor check efficiently.
    # Actually, the einsum above only checks the "valid" (i,j,k) combinations.
    # We need to ensure that for any (a,b,c) where c != k*n+i, the sum is 0.
    
    # Let's iterate over all a, b, c for correctness, using numpy for speed.
    # T_flat[a, b, c] = sum_r U_flat[a, r] V_flat[b, r] W_flat[c, r]
    
    # We can compute this as a batched operation or just loop over c for each a,b.
    # Since dimensions are small (6, 30, 20), we can do:
    # For each a in range(n*m):
    #   For each b in range(m*p):
    #     For each c in range(p*n):
    #       val = sum_r U[a,r] V[b,r] W[c,r]
    #       if valid(a,b,c): check val==1 else check val==0
    
    # To speed up, we can precompute U @ V^T? No, it's a 3-way product.
    # Let's use the matrix form:
    # M_ab[c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is equivalent to (U[a,:] * V[b,:]) dot W.T
    # Or M_ab = W @ (U[a,:] * V[b,:])^T
    
    for a in range(n * m):
        i = a // m
        j = a % m
        # Determine valid k for this (i,j)?
        # No, b determines j and k.
        # If b % p != k or b // p != j, then this (a,b) pair is structurally invalid?
        # Wait, the tensor is defined on the full spaces.
        # a = i*m+j. b = j'*p+k.
        # The term A[i,j] B[j',k] contributes to C[i,k] ONLY IF j == j'.
        # If j != j', the coefficient MUST be 0.
        
        for b in range(m * p):
            j_prime = b // p
            k = b % p
            
            # Compute vector v = U[a,:] * V[b,:]
            v = U_np[a, :] * V_np[b, :]
            # Compute M = W @ v
            M = W_np @ v
            
            for c in range(p * n):
                i_prime = c % n
                k_prime = c // n
                
                if i_prime == i and k_prime == k and j_prime == j:
                    if abs(M[c] - 1) > 1e-9:
                        return False
                else:
                    if abs(M[c]) > 1e-9:
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
    # Split the middle dimension m=3 into 2+1.
    # A = [A_left (2x2), A_right (2x1)]
    # B = [B_top (2x10); B_bottom (1x10)]
    # C = A_left @ B_top + A_right @ B_bottom
    
    # Term 2: A_right (2x1) @ B_bottom (1x10)
    # Rank = 2 * 1 * 10 = 20.
    
    # Term 1: A_left (2x2) @ B_top (2x10)
    # Split B_top into 5 blocks of 2x2.
    # Each 2x2 @ 2x2 can be done with Strassen (rank 7).
    # Total Rank = 20 + 5 * 7 = 55.
    
    # Let's construct this decomposition explicitly.
    
    cols_U = []
    cols_V = []
    cols_W = []
    
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
        
    # --- Term 2: A_right (2x1) @ B_bottom (1x10) ---
    # A_right indices in A (2x3): (0,2) -> idx 2, (1,2) -> idx 5
    # B_bottom indices in B (3x10): (2,k) -> idx 20+k
    # Output indices in C (2x10): (0,k) -> idx k*2+0, (1,k) -> idx k*2+1
    
    for k in range(10):
        # Mult 1: a_02 * b_2k -> c_0k
        add_col({2: 1}, {20+k: 1}, {k*2+0: 1})
        
        # Mult 2: a_12 * b_2k -> c_1k
        add_col({5: 1}, {20+k: 1}, {k*2+1: 1})
        
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
        # a = a00, b = a01, c = a10, d = a11
        # e = b0k0, f = b0k1, g = b1k0, h = b1k1
        
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
        
        # Indices:
        # U: a00=0, a01=1, a10=3, a11=4
        # V: b0k0=k0, b0k1=k1, b1k0=10+k0, b1k1=10+k1
        # W: C0k0=k0*2, C0k1=k1*2, C1k0=k0*2+1, C1k1=k1*2+1
        
        # m1 = (a00 + a11) * (b0k0 + b1k0)
        add_col({0:1, 4:1}, {k0:1, 10+k0:1}, {k0*2:1, k1*2+1:1})
        
        # m2 = (a10 + a11) * b0k0
        add_col({3:1, 4:1}, {k0:1}, {k0*2:-1, k0*2+1:1})
        
        # m3 = a00 * (b0k1 - b1k1)
        add_col({0:1}, {k1:1, 10+k1:-1}, {k1*2:1, k1*2+1:1})
        
        # m4 = a11 * (b1k0 - b1k1)
        add_col({4:1}, {10+k0:1, 10+k1:-1}, {k0*2:1, k1*2+1:-1})
        
        # m5 = (a00 + a01) * b1k1
        add_col({0:1, 1:1}, {10+k1:1}, {k1*2:1, k1*2+1:-1})
        
        # m6 = (a10 - a11) * (b0k0 + b0k1)
        add_col({3:1, 4:-1}, {k0:1, k1:1}, {k0*2+1:1, k1*2+1:-1})
        
        # m7 = (a01 - a11) * (b1k0 + b1k1)
        add_col({1:1, 4:-1}, {10+k0:1, 10+k1:1}, {k0*2:1, k1*2+1:-1})
        
    # Total rank = 20 + 5*7 = 55.
    
    rank_found = len(cols_U)
    U_mat = [[cols_U[c][r] for c in range(rank_found)] for r in range(n*m)]
    V_mat = [[cols_V[c][r] for c in range(rank_found)] for r in range(m*p)]
    W_mat = [[cols_W[c][r] for c in range(rank_found)] for r in range(p*n)]
    
    if _verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    # If verification fails, return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
