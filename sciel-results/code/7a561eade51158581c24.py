"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    start_time = time.time()
    deadline = 120.0
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def check_decomposition(U, V, W, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = sum(U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] for r in range(rank))
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(val - target) > 1e-6:
                            return False
        return True
    
    def to_half_int_list(arr):
        """Convert numpy array to list of lists with half-integer values."""
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                # Round to nearest half-integer
                rounded = round(val * 2) / 2.0
                if abs(val - rounded) < 1e-6:
                    if rounded == int(rounded):
                        row.append(int(rounded))
                    else:
                        row.append(rounded)
                else:
                    row.append(rounded)
            result.append(row)
        return result
    
    # Try ranks from 16 upward
    for rank in range(16, best_rank + 1):
        if time.time() - start_time > deadline - 20:
            break
        
        found = False
        # Use multiple random initializations
        for seed in range(20):
            if time.time() - start_time > deadline - 20:
                break
            
            np.random.seed(seed * 7919 + rank * 104729)
            
            # Initialize with small random values
            U0 = np.random.randn(n*m, rank) * 0.5
            V0 = np.random.randn(m*p, rank) * 0.5
            W0 = np.random.randn(p*n, rank) * 0.5
            
            def residuals(x):
                U = x[:n*m*rank].reshape(n*m, rank)
                V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                res = []
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                val = sum(U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] for r in range(rank))
                                target = 1.0 if j1 == j2 else 0.0
                                res.append(val - target)
                return np.array(res)
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=5000)
                if result.cost < 1e-10:
                    U = result.x[:n*m*rank].reshape(n*m, rank)
                    V = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Round to half-integers and check
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    if check_decomposition(U_rounded, V_rounded, W_rounded, rank):
                        best_rank = rank
                        best_result = {
                            "U": to_half_int_list(U_rounded),
                            "V": to_half_int_list(V_rounded),
                            "W": to_half_int_list(W_rounded)
                        }
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
