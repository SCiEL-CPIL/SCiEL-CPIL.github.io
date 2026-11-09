"""Combined program with linear system search for lower rank decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Baseline: Schoolbook Decomposition (Rank 60)
    rank_baseline = n * m * p
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1
    
    if time.time() - start_time > 5.0:
        return {"U": U_base, "V": V_base, "W": W_base}

    best_solution = None
    best_rank = rank_baseline
    deadline = start_time + time_limit - 5.0
    
    # Search for lower rank
    # We iterate R from 59 down to 40
    for R in range(rank_baseline - 1, 39, -1):
        if time.time() > deadline:
            break
            
        # Number of trials per rank
        # Start with more trials for higher ranks, decrease for lower
        trials = 1000 if R > 55 else 500
        
        for t in range(trials):
            if time.time() > deadline:
                break
                
            # Random U, V with entries from {-1, 0, 1}
            # Use a seed for reproducibility in debugging, but random for search
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            W_candidate = np.zeros((p * n, R), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Construct A of shape (m, R)
                    # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, R), dtype=int)
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A * w = 1
                    # We look for w in {0, 0.5, 1, ...}
                    # First, check if 1 is in the column space of A
                    # Use least squares
                    w_ls, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    
                    # Check if w_ls is close to half-integers
                    # If yes, we have a candidate
                    w_candidate = np.round(w_ls * 2) / 2.0
                    
                    # Verify A * w_candidate == 1
                    if np.allclose(A.dot(w_candidate), np.ones(m), atol=1e-9):
                        W_candidate[k * n + i, :] = w_candidate
                    else:
                        # Try to find a solution in the null space?
                        # If lstsq didn't yield a nice solution, maybe a different one does.
                        # For now, mark invalid to keep speed.
                        # We can improve this by checking if the system is consistent
                        # and then searching for a lattice point.
                        
                        # Check consistency: rank(A) == rank([A | 1])
                        A_aug = np.hstack((A, np.ones((m, 1))))
                        if np.linalg.matrix_rank(A) != np.linalg.matrix_rank(A_aug):
                            valid = False
                            break
                        else:
                            # System is consistent, but lstsq solution wasn't nice.
                            # This is a failure of our heuristic.
                            # We can try a few random steps in null space.
                            # Null space basis Z
                            # w = w_p + Z * y
                            # We want w in 0.5 * Z^R
                            # This is a lattice problem.
                            # For now, just mark invalid to avoid complex search.
                            valid = False
                            break
                            
                if not valid:
                    break
            
            if valid:
                # Verify the full tensor equation
                # Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
                valid_full = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(R):
                                s += U[i * m + j, c] * V[j * p + k, c] * W_candidate[k * n + i, c]
                            if abs(s - 1) > 1e-9:
                                valid_full = False
                                break
                        if not valid_full:
                            break
                    if not valid_full:
                        break
                
                if valid_full:
                    # Convert to lists
                    best_solution = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_candidate.tolist()
                    }
                    best_rank = R
                    # If we find a rank significantly lower, we can break early
                    # but we want the minimum, so we continue to lower R
                    # However, if R is very low, we might break to save time for verification?
                    # No, we just continue the outer loop to try R-1.
                    # But if we found R=50, we should check R=49 etc.
                    # The loop structure handles this.
                    # We can break the inner trial loop to try next R?
                    # No, we might find a better solution for the same R?
                    # Rank is the score, so any solution for R is equal to any other for R.
                    # So we can break the trial loop for this R.
                    break

    if best_solution is not None:
        return best_solution
    else:
        return {"U": U_base, "V": U_base and V_base and W_base or U_base, "V": V_base, "W": W_base}
