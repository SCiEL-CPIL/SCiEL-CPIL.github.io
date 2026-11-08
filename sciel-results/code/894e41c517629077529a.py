"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    # Schoolbook baseline: rank = n*m*p = 100
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        """Check if decomposition is valid"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Verify schoolbook
    if not verify(U, V, W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find a rank-99 decomposition (small improvement)
    # Strategy: try to merge two terms by adjusting coefficients
    # This is heuristic and may not succeed
    
    for target_rank in range(99, 94, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Initialize with first target_rank terms
            U_try = U[:, :target_rank].copy()
            V_try = V[:, :target_rank].copy()
            W_try = W[:, :target_rank].copy()
            
            # The remaining terms need to be absorbed - generally not possible
            # Try a different approach: random perturbation
            U_try = U[:, :target_rank].copy()
            V_try = V[:, :target_rank].copy()
            W_try = W[:, :target_rank].copy()
            
            # Add small random perturbations to try to absorb missing terms
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                
                if verify(U_try, V_try, W_try):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    found = True
                    break
                
                # Randomly adjust one entry
                mat = np.random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    r, c = np.random.randint(n * m), np.random.randint(target_rank)
                    U_try[r, c] += np.random.uniform(-0.1, 0.1)
                elif mat == 'V':
                    r, c = np.random.randint(m * p), np.random.randint(target_rank)
                    V_try[r, c] += np.random.uniform(-0.1, 0.1)
                else:
                    r, c = np.random.randint(p * n), np.random.randint(target_rank)
                    W_try[r, c] += np.random.uniform(-0.1, 0.1)
        
        if found:
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
