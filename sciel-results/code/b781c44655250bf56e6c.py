"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return False
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1
        return True
    
    def add_strassen(j1, j2, k1, k2):
        """
        Strassen rank-7 for A[:, j1:j2+1] * B[j1:j2+1, k1:k2+1]
        j1, j2 are the two j indices; k1, k2 are the two k indices.
        i ranges over 0, 1.
        """
        # M1 = (a_i,j1 + a_i,j2) * (b_j1,k1 + b_j2,k2)
        # Contributes to C_i,k1 and C_i,k2
        add_rank_step(
            [(u_idx(0, j1), 1), (u_idx(1, j1), 1),
             (u_idx(0, j2), 1), (u_idx(1, j2), 1)],
            [(v_idx(j1, k1), 1), (v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), 1), (w_idx(k1, 1), 1),
             (w_idx(k2, 0), 1), (w_idx(k2, 1), 1)]
        )
        
        # M2 = (a_1,j1 + a_1,j2) * b_j1,k1
        # Contributes to C_1,k1 (+) and C_1,k2 (-)
        add_rank_step(
            [(u_idx(1, j1), 1), (u_idx(1, j2), 1)],
            [(v_idx(j1, k1), 1)],
            [(w_idx(k1, 1), 1), (w_idx(k2, 1), -1)]
        )
        
        # M3 = a_0,j1 * (b_j1,k2 - b_j2,k2)
        # Contributes to C_0,k2 and C_1,k2
        add_rank_step(
            [(u_idx(0, j1), 1)],
            [(v_idx(j1, k2), 1), (v_idx(j2, k2), -1)],
            [(w_idx(k2, 0), 1), (w_idx(k2, 1), 1)]
        )
        
        # M4 = a_1,j2 * (b_j2,k1 - b_j1,k1)
        # Contributes to C_0,k1 and C_1,k1
        add_rank_step(
            [(u_idx(1, j2), 1)],
            [(v_idx(j2, k1), 1), (v_idx(j1, k1), -1)],
            [(w_idx(k1, 0), 1), (w_idx(k1, 1), 1)]
        )
        
        # M5 = (a_0,j1 + a_0,j2) * b_j2,k2
        # Contributes to C_0,k1 (-) and C_0,k2 (+)
        add_rank_step(
            [(u_idx(0, j1), 1), (u_idx(0, j2), 1)],
            [(v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), -1), (w_idx(k2, 0), 1)]
        )
        
        # M6 = (a_1,j1 - a_0,j1) * (b_j1,k1 + b_j1,k2)
        # Contributes to C_1,k2
        add_rank_step(
            [(u_idx(1, j1), 1), (u_idx(0, j1), -1)],
            [(v_idx(j1, k1), 1), (v_idx(j1, k2), 1)],
            [(w_idx(k2, 1), 1)]
        )
        
        # M7 = (a_0,j2 - a_1,j2) * (b_j2,k1 + b_j2,k2)
        # Contributes to C_0,k1
        add_rank_step(
            [(u_idx(0, j2), 1), (u_idx(1, j2), -1)],
            [(v_idx(j2, k1), 1), (v_idx(j2, k2), 1)],
            [(w_idx(k1, 0), 1)]
        )
    
    def add_schoolbook_2x1x2(j, k1, k2):
        """
        Schoolbook rank-4 for A[:, j] * B[j, k1:k2+1]
        """
        for i in range(n):
            for k in [k1, k2]:
                add_rank_step(
                    [(u_idx(i, j), 1)],
                    [(v_idx(j, k), 1)],
                    [(w_idx(k, i), 1)]
                )
    
    # Block 1: j=0, all k=0..5 -> 2x1x6, rank 12
    for i in range(n):
        for k in range(p):
            add_rank_step(
                [(u_idx(i, 0), 1)],
                [(v_idx(0, k), 1)],
                [(w_idx(k, i), 1)]
            )
    # col is now 12
    
    # Block 2: j={1,2}, k=0..5 -> 2x2x6, rank 21 (3 Strassen blocks)
    add_strassen(1, 2, 0, 1)
    add_strassen(1, 2, 2, 3)
    add_strassen(1, 2, 4, 5)
    # col is now 12 + 21 = 33
    
    # Block 3: j={3,4}, k=0..5 -> 2x2x6, rank 21 (3 Strassen blocks)
    add_strassen(3, 4, 0, 1)
    add_strassen(3, 4, 2, 3)
    add_strassen(3, 4, 4, 5)
    # col is now 33 + 21 = 54
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
