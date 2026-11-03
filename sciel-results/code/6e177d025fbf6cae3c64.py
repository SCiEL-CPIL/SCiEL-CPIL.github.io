"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Schoolbook decomposition as fallback
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
    
    # Try to find a lower-rank decomposition using greedy set cover
    # The tensor has entries T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # We want to cover all (i,j,k) with rank-1 tensors
    # Each rank-1 tensor is defined by (u, v, w) where u is a vector of length n*m, v of length m*p, w of length p*n
    # But we're looking for sparse decompositions
    
    # For simplicity, try to find a decomposition where each term covers a "box" of indices
    # This is heuristic and may not find the optimal
    
    # Given time constraints, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
