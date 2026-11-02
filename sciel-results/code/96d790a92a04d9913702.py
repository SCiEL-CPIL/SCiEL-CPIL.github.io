import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _build_rank_56(n, m, p):
    """
    Build the rank-56 decomposition using Strassen on 2x2 blocks.
    This is the standard optimal known simple construction for these dimensions.
    """
    total_rank = 56
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            a11_idx = 2 * u
            a12_idx = 2 * u + 1
            a21_idx = 4 + 2 * u
            a22_idx = 4 + 2 * u + 1
            
            b11_idx = 16 * u + 2 * v
            b12_idx = 16 * u + 2 * v + 1
            b21_idx = 16 * u + 8 + 2 * v
            b22_idx = 16 * u + 8 + 2 * v + 1
            
            c11_idx = 4 * v
            c12_idx = 4 * v + 2
            c21_idx = 4 * v + 1
            c22_idx = 4 * v + 3
            
            # Strassen 7 products for 2x2 matrices:
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank_56(n, m, p)
    return {"U": U, "V": V, "W": W}
