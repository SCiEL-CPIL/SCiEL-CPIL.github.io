import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 3, 3, 7
    
    # Strategy: Use standard decomposition for 3x3x7
    # Total terms: 3*3*7 = 63
    
    rank_total = 63
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    def add_term(i, j, k, val_u, val_v, val_w, col):
        """Add a rank-1 contribution to column col"""
        U[i * m + j][col] += val_u
        V[j * p + k][col] += val_v
        W[k * n + i][col] += val_w
    
    # Standard decomposition: one term for each (i,j,k) triple
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term(i, j, k, 1, 1, 1, col)
                col += 1
    
    # Verify rank
    assert col == rank_total, f"Expected rank {rank_total}, got {col}"
    
    return {"U": U, "V": V, "W": W}
