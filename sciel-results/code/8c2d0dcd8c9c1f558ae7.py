"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
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
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # For each candidate rank, try to find a valid decomposition
    for rank in range(rank_school - 1, 20, -1):
        if time.time() > deadline:
            break
        
        # Try random initialization and check
        # This is a heuristic; we check if a random rank-R decomposition works
        # (very unlikely, but worth trying a few times)
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-2, 3, size=(n*m, rank))
            V = np.random.randint(-2, 3, size=(m*p, rank))
            W = np.random.randint(-2, 3, size=(p*n, rank))
            
            # Check if this decomposition is valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k in range(p):
                            for k2 in range(p):
                                for i2 in range(n):
                                    lhs = 1 if (j == j2 and k == k2 and i == i2) else 0
                                    rhs = sum(U[i*m+j, r] * V[j2*p+k, r] * W[k*n+i2, r] for r in range(rank))
                                    if lhs != rhs:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best_result
# EVOLVE-BLOCK-END
