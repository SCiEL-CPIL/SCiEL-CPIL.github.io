"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Precompute the schoolbook as fallback
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_rank_with_patterns(target_rank, num_attempts=200):
        """Try to find a decomposition of given rank using structured patterns."""
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Generate random integer patterns for U and V
            # Use entries from {-1, 0, 1} with some structure
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            
            # For each column c, assign random sparse patterns
            for c in range(target_rank):
                # U column: random entries in {-1, 0, 1}
                U[:, c] = np.random.choice([-1, 0, 0, 0, 1], size=n * m)
                # V column: random entries in {-1, 0, 1}
                V[:, c] = np.random.choice([-1, 0, 0, 0, 1], size=m * p)
            
            # Now solve for W: for each (i, k), we need
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
            # This gives us a system of equations in W
            
            # For each output (i, k), we have m equations (one for each j)
            # with target_rank unknowns (W[k*n+i, c] for c in range(target_rank))
            
            # Build the linear system: A[i,k] @ w_vec = b
            # where A[i,k] is m x target_rank and b is m x 1
            
            W_solution = np.zeros((p * n, target_rank), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the m x target_rank matrix A
                    A = np.zeros((m, target_rank), dtype=float)
                    for j in range(m):
                        for c in range(target_rank):
                            A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    # Solve A @ w = ones(m)
                    # This is an underdetermined system (m=3 equations, target_rank unknowns)
                    # Use least squares with minimum norm solution
                    try:
                        w_vec, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    # Check if the solution is exact (residuals should be ~0)
                    if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-10:
                        valid = False
                        break
                    
                    W_solution[k * n + i, :] = w_vec
            
            if not valid:
                continue
            
            # Verify the full decomposition
            is_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U[i * m + j, c] * V[j * p + k, c] * W_solution[k * n + i, c]
                        if abs(s - 1.0) > 1e-9:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            
            if is_valid:
                # Check if entries are close to integers or half-integers
                # (they should be since we're solving a linear system with integer coefficients)
                U_list = [[round(U[i, c], 6) for c in range(target_rank)] for i in range(n * m)]
                V_list = [[round(V[i, c], 6) for c in range(target_rank)] for i in range(m * p)]
                W_list = [[round(W_solution[i, c], 6) for c in range(target_rank)] for i in range(p * n)]
                
                # Verify once more with the rounded values
                final_valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
                            if abs(s - 1.0) > 1e-9:
                                final_valid = False
                                break
                        if not final_valid:
                            break
                    if not final_valid:
                        break
                
                if final_valid:
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from below schoolbook down to a minimum
    # Theoretical lower bound for 3x3x7 is at least 3*3*7/2 = 47.25, so >= 48
    # But let's be more conservative and start from 60
    
    for target_rank in range(60, 47, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank_with_patterns(target_rank, num_attempts=100)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found a valid decomposition, stop
    
    if best_result is None:
        best_result = make_schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
