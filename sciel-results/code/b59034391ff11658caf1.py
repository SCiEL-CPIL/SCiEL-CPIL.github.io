import time
import numpy as np
import random

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def check_consistency(U, V, R):
        """
        Check if for all i, k, the system M^{(i,k)} w = 1 is consistent.
        Returns W if consistent, else None.
        """
        W = [[0] * R for _ in range(pn)]
        for i in range(n):
            for k in range(p):
                # Construct M (4 x R)
                # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                M = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        M[j, r] = U[i*m + j][r] * V[j*p + k][r]
                
                # Solve M w = 1
                # Use lstsq
                try:
                    w, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    # Check residual
                    if residuals is not None and residuals[0] > 1e-9:
                        return None
                    # Assign W[k*n+i, r] = w[r]
                    for r in range(R):
                        W[k*n + i][r] = w[r]
                except:
                    return None
        return W

    def random_search(R, num_trials=1000):
        for _ in range(num_trials):
            if time.time() > deadline - 5.0:
                return None
            
            # Random U, V with small integers
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(R)] for _ in range(nm)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(R)] for _ in range(mp)]
            
            W = check_consistency(U, V, R)
            if W is not None:
                # Verify full decomposition
                if verify_decomposition(U, V, W, R):
                    return {"U": U, "V": V, "W": W}
        return None

    best_result = schoolbook()
    best_rank = n * m * p
    
    min_possible_rank = max(nm, mp, pn)
    
    # Try to find lower rank
    for R in range(min_possible_rank, best_rank):
        if time.time() > deadline - 10.0:
            break
        
        # Increase trials for lower ranks
        trials = 2000 if R < 35 else 500
        
        result = random_search(R, trials)
        if result is not None:
            best_result = result
            best_rank = R
            break
            
    return best_result
