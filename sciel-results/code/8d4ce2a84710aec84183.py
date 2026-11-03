import numpy as np
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for a in range(dim_u):
        i_a = a // m
        j_a = a % m
        for b in range(dim_v):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(dim_w):
                k_c = c // n
                i_c = c % n
                if i_a == i_c and k_b == k_c:
                    T[a, b, c] = 1.0

    # Precompute indices for efficiency
    # We want to solve M_c * W_c = T_c for each c
    # M_c is (dim_u * dim_v) x R
    # M_c[a*dim_v+b, r] = U[a, r] * V[b, r]
    
    # We try ranks from 26 to 29
    for R in range(26, 30):
        # Try sparse random initializations
        # Each row of U and V has exactly 2 or 3 non-zero entries
        for nnz in [2, 3]:
            for trial in range(500):
                # Generate sparse U
                U = np.zeros((dim_u, R), dtype=float)
                V = np.zeros((dim_v, R), dtype=float)
                
                # Randomly place non-zeros
                # Ensure each column has some non-zeros
                for r in range(R):
                    # Pick nnz rows for U
                    idx_u = np.random.choice(dim_u, nnz, replace=False)
                    for idx in idx_u:
                        U[idx, r] = np.random.choice([-1, 1])
                        
                    # Pick nnz rows for V
                    idx_v = np.random.choice(dim_v, nnz, replace=False)
                    for idx in idx_v:
                        V[idx, r] = np.random.choice([-1, 1])
                
                # Solve for W
                W = np.zeros((dim_w, R))
                valid = True
                for c in range(dim_w):
                    T_c_flat = T[:, :, c].flatten()
                    # Construct M
                    # M[i, r] = U[i, r] * V[j, r] where i = a*dim_v+b
                    # This can be done as:
                    # U_expanded = U[:, :, None] * V[None, :, :] ? No.
                    # M is (dim_u*dim_v, R)
                    # M[a*dim_v+b, r] = U[a, r] * V[b, r]
                    # We can form this by broadcasting
                    U_col = U[:, :, None] # (dim_u, R, 1)
                    V_row = V[None, :, :] # (1, dim_v, R)
                    # Product shape (dim_u, dim_v, R)
                    M_3d = U_col * V_row
                    M = M_3d.reshape(-1, R)
                    
                    try:
                        W_c, residuals, rank, s = np.linalg.lstsq(M, T_c_flat, rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check for integer solution
                    W_c_int = np.round(W_c)
                    if not np.allclose(W_c, W_c_int, atol=1e-6):
                        valid = False
                        break
                        
                    W[c, :] = W_c_int
                
                if valid:
                    # Verify
                    # Reconstruct T
                    # T_prime[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                    # Vectorized check
                    # Sum over r of U[a, r] * V[b, r] * W[c, r]
                    # This is a tensor contraction
                    # Let's do it loop-free for speed if possible, or just loop c
                    # For c in range(dim_w):
                    #   T_prime[:, :, c] = (U @ V.T) * W[c, :] ... No.
                    #   T_prime[a, b, c] = dot(U[a], V[b] * W[c])
                    
                    # Efficient check:
                    # Compute P = U @ V.T (dim_u x dim_v)
                    # Then T_prime[a, b, c] = sum_r P[a, b, r?] No.
                    # P[a, b] is not enough because W depends on c and r.
                    # We need sum_r U[a, r] V[b, r] W[c, r].
                    # This is element-wise product of (U V^T) with something? No.
                    # It is the sum of outer products.
                    
                    # Let's just verify with a loop, it's small (10*15*10 = 1500 entries)
                    T_prime = np.zeros_like(T)
                    for a in range(dim_u):
                        for b in range(dim_v):
                            for c in range(dim_w):
                                T_prime[a, b, c] = np.dot(U[a, :], V[b, :] * W[c, :])
                    
                    if np.allclose(T_prime, T, atol=1e-6):
                        return {
                            "U": U.astype(int).tolist(),
                            "V": V.astype(int).tolist(),
                            "W": W.astype(int).tolist()
                        }
                        
    # Fallback to schoolbook rank 30
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
