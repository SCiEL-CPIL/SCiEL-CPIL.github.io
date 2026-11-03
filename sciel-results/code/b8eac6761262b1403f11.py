import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T as a 3D array
    # T[i, j, k] = 1 if it's a valid multiplication term
    # But we need to map to the flattened indices.
    # T_flat[alpha, beta, gamma]
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    
    # We will work with the unfolded matrices.
    # T1[alpha, beta*pn + gamma]
    
    # Initialize best solution
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Try to find lower rank
    for R in range(best_rank - 1, 60, -1):
        if time.time() > deadline:
            break
            
        # Random initialization
        np.random.seed(42 + R) # Reproducible for testing, but maybe random is better?
        # Use small integers
        U = np.random.randint(-1, 2, (nm, R))
        V = np.random.randint(-1, 2, (mp, R))
        W = np.random.randint(-1, 2, (pn, R))
        
        # Iterate
        for it in range(50):
            if time.time() > deadline:
                break
                
            # Update W
            # For each gamma, solve A_gamma W_gamma = b_gamma
            # A_gamma[alpha * mp + beta, c] = U[alpha, c] * V[beta, c]
            # b_gamma[alpha * mp + beta] = T[alpha, beta, gamma]
            
            # Precompute T slices
            # This is slow in Python, so we vectorize if possible.
            
            # Let's just check error and break if 0.
            # Error calculation:
            # Sum_c U[alpha, c] V[beta, c] W[gamma, c]
            
            # This is O(nm * mp * pn * R)
            # 72 * 24 * 18 * 70 ~ 2 million ops. Fast enough.
            
            # Compute reconstructed tensor
            # T_recon[alpha, beta, gamma] = sum_c U[alpha, c] V[beta, c] W[gamma, c]
            
            # We can do this efficiently using einsum or matmul.
            # T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
            
            T_recon = np.einsum('ac,bc,dc->abd', U, V, W, optimize=True)
            
            # Target T
            # T_target[alpha, beta, gamma] = 1 if alpha=(i,j), beta=(j,k), gamma=(k,i)
            # else 0
            
            # We can precompute T_target.
            # T_target = np.zeros((nm, mp, pn))
            # for i in range(n):
            #   for j in range(m):
            #     for k in range(p):
            #       alpha = i*m + j
            #       beta = j*p + k
            #       gamma = k*n + i
            #       T_target[alpha, beta, gamma] = 1
            
            # Check error
            if np.allclose(T_recon, T_target, atol=1e-9):
                best_rank = R
                best_U = U
                best_V = V
                best_W = W
                break
                
            # Update W
            # For each gamma, solve least squares
            # We can vectorize this update.
            # A[alpha, beta, c] = U[alpha, c] V[beta, c]
            # We want to minimize || A W - T ||_F^2
            # This is equivalent to solving for each gamma.
            
            # Let's do a simple gradient step or Gauss-Newton?
            # Given time constraints, let's just try random restarts.
            
        if best_rank < best_rank:
            break

    if best_U is None:
        # Fallback to schoolbook
        U = np.zeros((nm, best_rank))
        V = np.zeros((mp, best_rank))
        W = np.zeros((pn, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
