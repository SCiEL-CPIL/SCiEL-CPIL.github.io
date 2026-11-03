import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Construct a rank-42 tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    Strategy: Partition A (2x4) into two 2x2 blocks A_left, A_right.
    Partition B (4x6) into two 2x6 blocks B_top, B_bot.
    C = A_left * B_top + A_right * B_bot
    
    Each 2x6 block is split into three 2x2 column blocks.
    So we have 6 independent 2x2 * 2x2 multiplications.
    Each uses Strassen's rank-7 algorithm.
    Total rank = 6 * 7 = 42.
    """
    start_time = time.time()
    deadline = 145.0  # 150s limit, keep 5s margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    rank = 42
    U = np.zeros((n * m, rank))  # (8, 42)
    V = np.zeros((m * p, rank))  # (24, 42)
    W = np.zeros((p * n, rank))  # (12, 42)
    
    # Strassen terms: (u_factors, v_factors, w_factors)
    # Each factor is a list of ((local_row, local_col), coeff)
    strassen_terms = [
        # M1
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        # M2
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        # M3
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        # M4
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        # M5
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        # M6
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        # M7
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    col_idx = 0
    for source in range(2):  # 0=A_left, 1=A_right
        for c in range(3):  # column block
            for t in range(7):
                if time.time() > deadline:
                    break
                u_factors, v_factors, w_factors = strassen_terms[t]
                
                for (pi, pj), coeff in u_factors:
                    u_row = pi * m + (pj + 2 * source)
                    U[u_row, col_idx] += coeff
                
                for (qi, qj), coeff in v_factors:
                    v_row = (qi + 2 * source) * p + (2 * c + qj)
                    V[v_row, col_idx] += coeff
                
                for (ri, rj), coeff in w_factors:
                    w_row = (2 * c + rj) * n + ri
                    W[w_row, col_idx] += coeff
                
                col_idx += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
