import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p), dtype=float)
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Lower bound on rank
    # Rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Try ranks from 30 upwards
    
    best_result = None
    best_rank = rank_school

    for r in range(30, 60):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations of U and V
        for restart in range(500):
            if time.time() > deadline:
                break
                
            # Initialize U and V with small integers
            # Using {-1, 0, 1} to keep values small
            U = np.random.randint(-1, 2, size=(n, m, r))
            V = np.random.randint(-1, 2, size=(m, p, r))
            
            # Solve for W
            W = np.zeros((p, n, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Construct M[i,k] of shape (m, r)
                    # M[j, c] = U[i, j, c] * V[j, k, c]
                    M = U[i, :, :] * V[:, k, :]
                    
                    # Solve M @ w = ones(m)
                    # M is (m, r), w is (r,)
                    # Use lstsq
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except Exception:
                        valid = False
                        break
                    
                    W[k, i, :] = w
            
            if not valid:
                continue
                
            # Check if W is close to half-integers
            W_check = np.round(W * 2) / 2
            
            if np.allclose(W, W_check, atol=1e-6):
                # Verify the decomposition
                # prod[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                prod = np.einsum('ijc,jkc,kic->ijk', U, V, W_check, optimize=True)
                
                if np.allclose(prod, T, atol=1e-6):
                    # Found a valid decomposition
                    if r < best_rank:
                        best_rank = r
                        best_result = {
                            "U": U.reshape(n*m, r).tolist(),
                            "V": V.reshape(m*p, r).tolist(),
                            "W": W_check.reshape(p*n, r).tolist()
                        }
                        break # Found solution at this rank, break restarts
        
        if best_result is not None and best_rank == r:
            break

    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
