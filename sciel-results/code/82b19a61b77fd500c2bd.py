import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _search_low_rank(n, m, p, max_rank, time_budget):
    """Search for a low-rank decomposition."""
    start_time = time.time()
    best_result = None
    best_rank = max_rank + 1
    
    nmp = n * m * p
    num_eq = nmp  # 96 equations
    
    for R in range(1, max_rank + 1):
        if time.time() - start_time > time_budget:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() - start_time > time_budget:
                break
            
            # Initialize with small random half-integers
            np.random.seed(R * 1000 + trial)
            U = np.random.randint(-2, 3, (n*m, R)) * 0.5
            V = np.random.randint(-2, 3, (m*p, R)) * 0.5
            W = np.random.randint(-2, 3, (p*n, R)) * 0.5
            
            # Check if this random initialization works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(R):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1) > 1e-10:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Convert to list of lists with half-integers
                U_list = [[float(U[i,j]) for j in range(R)] for i in range(n*m)]
                V_list = [[float(V[i,j]) for j in range(R)] for i in range(m*p)]
                W_list = [[float(W[i,j]) for j in range(R)] for i in range(p*n)]
                
                if best_rank > R:
                    best_rank = R
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 5 second safety margin
    
    # Try to find a low-rank decomposition
    max_search_rank = 95  # Try ranks 1 to 95
    time_budget = 30  # Spend up to 30 seconds on search
    
    result, rank = _search_low_rank(n, m, p, max_search_rank, time_budget)
    
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
