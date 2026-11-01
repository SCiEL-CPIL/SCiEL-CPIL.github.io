import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 2, 2, 2
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 scalar products
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # W convention: W[k*n + i] where C[k][i] is the element
    # C indices: 00=0, 10=1, 01=2, 11=3
    
    # Each term is a list of (U_indices, V_indices, W_indices_with_signs)
    # U_indices: list of (row_idx, coeff)
    # V_indices: list of (row_idx, coeff)
    # W_indices: list of (row_idx, coeff)
    
    terms = [
        # M1: (a00 + a11) * (b00 + b11)
        # c00 += m, c11 += m
        [ [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)] ],
        
        # M2: (a10 + a11) * b00
        # c10 += m, c11 -= m
        [ [(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)] ],
        
        # M3: a00 * (b01 - b11)
        # c01 += m, c11 += m
        [ [(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)] ],
        
        # M4: a11 * (b10 - b00)
        # c00 += m, c10 += m
        [ [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)] ],
        
        # M5: (a00 + a01) * b11
        # c00 -= m, c01 += m
        [ [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)] ],
        
        # M6: (a10 - a00) * (b00 + b01)
        # c11 += m
        [ [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)] ],
        
        # M7: (a01 - a11) * (b10 + b11)
        # c00 += m
        [ [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)] ]
    ]
    
    for k in range(rank):
        u_data, v_data, w_data = terms[k]
        for idx, coeff in u_data:
            U[idx][k] += coeff
        for idx, coeff in v_data:
            V[idx][k] += coeff
        for idx, coeff in w_data:
            W[idx][k] += coeff

    return {"U": U, "V": V, "W": W}
