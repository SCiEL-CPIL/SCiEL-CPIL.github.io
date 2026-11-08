import time
import sys

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We use the schoolbook decomposition of the 6x6x6 tensor,
    # padded to cover the 4x5x5x6 problem.
    # Rank is 6*6*6 = 216? No, schoolbook rank is n*m*p.
    # For 6x6, rank is 36.
    # We iterate over i, j, k in 0..5.
    # If indices are within 4x5, 5x6, 6x4 bounds, we set the factor.
    
    max_dim = max(n, m, p)
    rank = max_dim * max_dim * max_dim
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(max_dim):
        for j in range(max_dim):
            for k in range(max_dim):
                # Check if this term is relevant to the 4x5x5x6 product
                # A is 4x5, so i < 4 and j < 5
                # B is 5x6, so j < 5 and k < 6
                # C is 4x6, so i < 4 and k < 6
                
                if i < n and j < m:
                    u_row = i * m + j
                    U[u_row][col] = 1
                if j < m and k < p:
                    v_row = j * p + k
                    V[v_row][col] = 1
                if k < p and i < n:
                    w_row = k * n + i
                    W[w_row][col] = 1
                
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
