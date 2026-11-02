import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Build the target tensor T
    # T is a 3D array of shape (U_dim, V_dim, W_dim)
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Function to verify a decomposition
    def verify(U, V, W, tol=1e-6):
        # U: (U_dim, R), V: (V_dim, R), W: (W_dim, R)
        R = U.shape[1]
        # Compute sum_r U[:,r] * V[:,r] * W[:,r]
        # This is a 3-way tensor contraction.
        # We can do this by iterating r or using einsum.
        # einsum('ar,br,cr->abc', U, V, W)
        try:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            if np.allclose(T_recon, T, atol=tol):
                return True
        except Exception:
            pass
        return False

    # Function to round to half-integers
    def round_half(x):
        return np.round(x * 2) / 2

    # Try ranks from low to high
    # Lower bound: max(U_dim, V_dim, W_dim)? No.
    # For matmul, rank >= max(n*m, m*p, p*n) is not true.
    # Rank >= max(n, m, p) is a loose bound.
    # Let's try from 20 to 48.
    
    best_result = None
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((U_dim, rank_school))
    V_school = np.zeros((V_dim, rank_school))
    W_school = np.zeros((W_dim, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    if time.time() > deadline - 10:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    # Heuristic: Try to find a rank < 48
    # We'll use a simple greedy approach or ALS.
    # Given the small size, we can try a few random restarts for specific ranks.
    
    target_ranks = [30, 32, 34, 36, 38, 40, 42, 44, 46]
    
    for R in target_ranks:
        if time.time() > deadline - 15:
            break
            
        # Try a few random initializations
        found = False
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 10:
                break
                
            np.random.seed(seed)
            # Initialize with small random values
            U = np.random.randn(U_dim, R) * 0.1
            V = np.random.randn(V_dim, R) * 0.1
            W = np.random.randn(W_dim, R) * 0.1
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Fix U, V. Solve for W.
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For each c, T[:,:,c] = sum_r (U[:,r] * V[:,r]) W[r,c]
                # Let M_r = U[:,r] * V[:,r] (outer product)
                # T_flat (U_dim*V_dim, W_dim) = sum_r M_r_flat (U_dim*V_dim, 1) * W[r, :] (1, W_dim)
                # This is not a simple matrix multiply because M_r depends on r.
                # We can write:
                # T[a,b,c] = sum_r (U[a,r] V[b,r]) W[c,r]
                # Let's vectorize over a,b.
                # For each r, define A_r = U[:,r] * V[:,r] (size U_dim x V_dim)
                # Then T[:,:,c] = sum_r A_r * W[r,c]
                # This is a linear system in W if we flatten T and A.
                # But A_r is different for each r.
                # We can construct a large matrix B of size (U_dim*V_dim) x R
                # where B[:, r] = A_r.ravel()
                # Then T.ravel() (size U_dim*V_dim*W_dim) is NOT directly B @ W.
                # T[a,b,c] = sum_r B[(a,b), r] * W[c, r]
                # This is a 3D contraction.
                # We can solve for each c separately.
                # For a fixed c, T[:,:,c] (U_dim x V_dim) = sum_r A_r * W[r,c]
                # Flatten T[:,:,c] to vector t_c (U_dim*V_dim)
                # t_c = sum_r A_r_flat * W[r,c]
                # This is: t_c = B @ w_c, where B is (U_dim*V_dim, R) and w_c is (R, 1)
                # So w_c = B^+ t_c (least squares)
                
                # Construct B: (U_dim*V_dim, R)
                # B[(a,b), r] = U[a,r] * V[b,r]
                # We can use broadcasting:
                # U (U_dim, R), V (V_dim, R)
                # U[:, :, None] * V[None, :, :] -> (U_dim, V_dim, R)
                # Reshape to (U_dim*V_dim, R)
                B = (U[:, :, None] * V[None, :, :]).reshape(U_dim * V_dim, R)
                
                # T_flat: (U_dim*V_dim, W_dim)
                T_flat = T.reshape(U_dim * V_dim, W_dim)
                
                # Solve B @ W_new.T = T_flat.T ?
                # T_flat[a,b, c] = sum_r B[a,b, r] W_new[r, c]
                # So T_flat = B @ W_new.T
                # W_new.T = B^+ T_flat
                # W_new = (B^+ T_flat).T
                
                try:
                    # Use lstsq
                    W_new, residuals, rank, sv = np.linalg.lstsq(B, T_flat.T, rcond=None)
                    W = W_new.T
                except Exception:
                    pass
                
                # Similar for U and V
                # Fix V, W. Solve for U.
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed a: T[a,:,:] = sum_r U[a,r] (V[:,r] * W[:,r])
                # Let C_r = V[:,r] * W[:,r] (V_dim x W_dim)
                # T[a,:,:] = sum_r U[a,r] C_r
                # Flatten T[a,:,:] to t_a (V_dim*W_dim)
                # t_a = sum_r U[a,r] C_r_flat
                # t_a = C @ u_a, where C is (V_dim*W_dim, R)
                C = (V[:, :, None] * W[None, :, :]).reshape(V_dim * W_dim, R)
                T_flat_a = T.reshape(U_dim, V_dim * W_dim).T # (V_dim*W_dim, U_dim)
                try:
                    U_new, residuals, rank, sv = np.linalg.lstsq(C, T_flat_a, rcond=None)
                    U = U_new.T
                except Exception:
                    pass
                
                # Fix U, W. Solve for V.
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed b: T[:,b,:] = sum_r V[b,r] (U[:,r] * W[:,r])
                # Let D_r = U[:,r] * W[:,r] (U_dim x W_dim)
                # T[:,b,:] = sum_r V[b,r] D_r
                # Flatten to t_b (U_dim*W_dim)
                # t_b = sum_r V[b,r] D_r_flat
                # t_b = D @ v_b
                D = (U[:, :, None] * W[None, :, :]).reshape(U_dim * W_dim, R)
                T_flat_b = T.reshape(V_dim, U_dim * W_dim).T # (U_dim*W_dim, V_dim)
                try:
                    V_new, residuals, rank, sv = np.linalg.lstsq(D, T_flat_b, rcond=None)
                    V = V_new.T
                except Exception:
                    pass
            
            # Round and verify
            U_r = round_half(U)
            V_r = round_half(V)
            W_r = round_half(W)
            
            if verify(U_r, V_r, W_r):
                # Check if we can reduce rank by removing zero columns
                # For simplicity, just return this rank
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
    # If no improvement found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
