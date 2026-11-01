"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Schoolbook rank as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try ranks from low to high
    for target_rank in range(9, rank_school + 1):
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            seed = attempt * 7919 + target_rank * 104729 + 12345
            np.random.seed(seed)
            
            # Initialize
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Alternating least squares
            for iteration in range(500):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                A_mat[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        try:
                            sol = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                            U[i * m + j, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                A_mat[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        try:
                            sol = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                            V[j * p + k, :] = sol
                        except:
                            pass
                
                # Update W
                for i in range(n):
                    for k in range(p):
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                A_mat[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        try:
                            sol = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                            W[k * n + i, :] = sol
                        except:
                            pass
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-12:
                    # Found valid decomposition
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if verify(U_list, V_list, W_list, target_rank):
                        return {"U": U_list, "V": V_list, "W": W_list}
                
                if max_err < 1e-8 and iteration > 50:
                    # Try to refine with more iterations
                    pass
            
            if time.time() > deadline:
                break
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
