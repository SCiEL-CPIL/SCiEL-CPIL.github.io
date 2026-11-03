import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the target tensor
    # T[w, u, v] = 1 if u = i*m+j, v = j*p+k, w = k*n+i for some i,j,k
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[w, u, v] = 1
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_schoolbook
    
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # Try to find lower rank decompositions
    # Known lower bounds for matrix multiplication:
    # For 3x5 * 5x6, the rank is at least n*p + m*p + n*m - n - m - p + 1 (not always tight)
    # Actually, let's try ranks from schoolbook-1 down to some lower bound
    
    # Theoretical lower bound: max(n*p, m*p, n*m) but that's not tight
    # Let's try a range of ranks
    
    # We'll use a randomized approach:
    # 1. Randomly initialize U, V with small integers/half-integers
    # 2. Solve for W via least squares
    # 3. Round U, V, W to half-integers
    # 4. Verify exactness
    
    # But this is very hard. Instead, let's try structured approaches.
    
    # Approach: Use the fact that matrix multiplication tensor has a certain structure.
    # We can try to find a decomposition where U, V, W have sparse structure.
    
    # Let's try ranks from 90 down to 70 (or lower if time permits)
    # For each rank, try multiple random initializations
    
    for R in range(rank_schoolbook - 1, 60, -1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        
        # Try a few random initializations
        num_attempts = 20 if R > 80 else 10
        
        for attempt in range(num_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization with small integers
            # Use values from {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}
            scale = 1.0
            U_curr = np.random.choice([-scale, 0, scale], size=(n * m, R))
            V_curr = np.random.choice([-scale, 0, scale], size=(m * p, R))
            
            # For some attempts, use half-integers
            if attempt % 3 == 0:
                U_curr = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, R))
                V_curr = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, R))
            
            # Build the matrix A where A[:, r] = vec(U_curr[:, r] * V_curr[:, r]^T)
            A = np.zeros((n * m * m * p, R))
            for r in range(R):
                M_r = np.outer(U_curr[:, r], V_curr[:, r])
                A[:, r] = M_r.flatten()
            
            # Solve for W using least squares
            # T_vec[w] = A @ W[w]
            W_solution = np.zeros((p * n, R))
            residual = 0.0
            
            for w in range(p * n):
                t_vec = T[w].flatten()
                W_w, residuals, rank_A, s = np.linalg.lstsq(A, t_vec, rcond=None)
                W_solution[w] = W_w
                residual += np.sum((A @ W_w - t_vec) ** 2)
            
            # Check if residual is small
            if residual < 1e-8:
                # Round to half-integers
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_solution * 2) / 2
                
                # Verify exactness
                T_recon = np.zeros((p * n, n * m, m * p))
                for w in range(p * n):
                    for r in range(R):
                        if abs(W_rounded[w, r]) > 1e-12:
                            T_recon[w] += W_rounded[w, r] * np.outer(U_rounded[:, r], V_rounded[:, r])
                
                if np.max(np.abs(T_recon - T)) < 1e-9:
                    best_rank = R
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    found = True
                    break
        
        if found:
            break
    
    # If we found a better solution, use it; otherwise use schoolbook
    if best_rank < rank_schoolbook:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
