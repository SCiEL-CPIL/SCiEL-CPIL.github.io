import time
import random

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42

    # Trivial decomposition
    U = [[0] * trivial_rank for _ in range(n * m)]
    V = [[0] * trivial_rank for _ in range(m * p)]
    W = [[0] * trivial_rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_rank = trivial_rank
    best_result = {"U": U, "V": V, "W": W}

    # Try to find a lower rank decomposition using random search with rounding
    # This is a heuristic approach that may find a better decomposition
    # if one exists with integer/half-integer entries
    
    # For small dimensions, we can try random projections and check if they
    # can be rounded to valid decompositions
    
    # First, let's try a structured approach based on Winograd-style schemes
    # For 2x3x7, we can try to exploit the structure
    
    # Try rank 35 (removing 7 terms)
    # Try rank 30 (removing 12 terms)
    # etc.
    
    # Given the time constraints, let's try a few specific ranks
    for target_rank in range(35, trivial_rank):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of rank target_rank
        # Using random initialization and iterative refinement
        
        # Initialize with random values
        U_r = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
        V_r = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
        W_r = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
        
        # Check if this decomposition is valid
        # For each (i, j, k), we need: sum_t U[i*m+j][t] * V[j*p+k][t] == W[k*n+i][t] * (something)
        # Actually, the tensor decomposition means:
        # T[i,j,k] = sum_t U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
        # where T[i,j,k] = 1 for all i,j,k (since we're decomposing the multiplication tensor)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(target_rank):
                        s += U_r[i * m + j][t] * V_r[j * p + k][t] * W_r[k * n + i][t]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            break

    return best_result
