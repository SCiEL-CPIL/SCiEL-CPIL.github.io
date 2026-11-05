import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin (limit is 150s)
    
    # Schoolbook solution as fallback (rank 48)
    target_rank = n * m * p  # 48
    U_school = [[0] * target_rank for _ in range(n * m)]
    V_school = [[0] * target_rank for _ in range(m * p)]
    W_school = [[0] * target_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find a lower rank decomposition
    # Known lower bound: max(n*p, m*p, n*m) = max(12, 16, 12) = 16
    # For 3x4x4, the rank is known to be around 18-23
    
    # Strategy: Use structured random search with {-1, 0, 1} entries
    # For each rank R, try random U and V, then solve for W
    
    for R in range(16, target_rank):
        if time.time() > deadline - 15.0:
            break
            
        # Number of random trials per rank
        # More trials for smaller R since they're harder
        trials = 5000 if R <= 25 else 2000
        
        for trial in range(trials):
            if time.time() > deadline - 5.0:
                break
                
            # Randomly initialize U and V with entries in {-1, 0, 1}
            # Shapes: U is (n*m, R), V is (m*p, R)
            U_flat = np.random.randint(-1, 2, size=(n * m, R))
            V_flat = np.random.randint(-1, 2, size=(m * p, R))
            
            # Reshape to (n, m, R) and (m, p, R)
            U = U_flat.reshape(n, m, R)
            V = V_flat.reshape(m, p, R)
            
            W_solution = np.zeros((p, n, R))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Construct matrix A of size (m, R)
                    # A[j, c] = U[i, j, c] * V[j, k, c]
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i, j, :] * V[j, k, :]
                    
                    # Solve A * W_vec = 1
                    # W_vec is (R,)
                    try:
                        # Use lstsq to handle underdetermined systems
                        W_vec, residuals, rank, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        
                        # Check if the solution is exact
                        if not np.allclose(A @ W_vec, np.ones(m), atol=1e-9):
                            valid = False
                            break
                            
                        # Check if entries are close to integers or half-integers
                        W_scaled = 2 * W_vec
                        if not np.allclose(W_scaled, np.round(W_scaled), atol=1e-9):
                            # Try to snap to nearest half-integer
                            W_snapped = np.round(W_scaled) / 2.0
                            if not np.allclose(A @ W_snapped, np.ones(m), atol=1e-9):
                                valid = False
                                break
                            W_vec = W_snapped
                            
                        W_solution[k, i, :] = W_vec
                        
                    except Exception:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # If we get here, we have a valid decomposition of rank R
            # Final verification
            is_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(R):
                            val += U[i, j, c] * V[j, k, c] * W_solution[k, i, c]
                        if abs(val - 1.0) > 1e-9:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            
            if is_valid:
                # Convert to list of lists
                U_out = U_flat.tolist()
                V_out = V_flat.tolist()
                W_out = W_solution.transpose(1, 0, 2).reshape(p * n, R).tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
                
    return fallback
