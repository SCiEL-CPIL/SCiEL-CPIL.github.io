"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 30
    pn = p * n  # 20
    
    # The tensor constraint: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # 
    # Number of constraints: n*m*p = 60
    # Variables: rank * (nm + mp + pn) = rank * 56
    #
    # We need rank such that we can satisfy 60 equations.
    # Minimum possible rank: at least max(nm, mp, pn) / something, but let's try small ranks.
    
    # Try ranks from a lower bound upward
    # Lower bound: the tensor has rank at least max(n, m, p) for standard matrix mult,
    # but this is a different tensor. Let's just try small ranks.
    
    best_result = None
    best_rank = None
    
    # Try various ranks
    for rank in range(10, 61):
        if time.time() > deadline:
            break
        
        # Random initialization with small integers
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            random.seed(rank * 1000 + attempt)
            
            # Initialize with random small integers (-1, 0, 1)
            U = np.zeros((nm, rank), dtype=float)
            V = np.zeros((mp, rank), dtype=float)
            W = np.zeros((pn, rank), dtype=float)
            
            # Random sparse initialization
            for c in range(rank):
                # Randomly assign non-zero entries
                u_rows = random.sample(range(nm), min(nm, random.randint(1, 3)))
                for r in u_rows:
                    U[r, c] = random.choice([-1, 1])
                
                v_rows = random.sample(range(mp), min(mp, random.randint(1, 3)))
                for r in v_rows:
                    V[r, c] = random.choice([-1, 1])
                
                w_rows = random.sample(range(pn), min(pn, random.randint(1, 3)))
                for r in w_rows:
                    W[r, c] = random.choice([-1, 1])
            
            # Check if this satisfies the constraints
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
            residual = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        residual += abs(s - 1.0)
            
            if residual < 1e-9:
                # Found a valid decomposition!
                best_rank = rank
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
