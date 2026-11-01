import numpy as np
import time

def construct():
    """
    Returns a rank-14 exact tensor decomposition for 2x2 by 2x4 matrix multiplication.
    U: (n*m, rank) = (4, 14)
    V: (m*p, rank) = (8, 14)
    W: (p*n, rank) = (8, 14)
    """
    n, m, p = 2, 2, 4
    rank = 14
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B local indices: 00->0, 01->1, 10->2, 11->3
    # C local indices: 00->0, 01->1, 10->2, 11->3
    
    # Strassen terms: (u_indices, v_indices, w_indices)
    strassen_terms = [
        # M1 = (a00 + a11) * (b00 + b11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (a10 + a11) * b00
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # M3 = a00 * (b01 - b11)
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # M4 = a11 * (b10 - b00)
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M5 = (a00 + a01) * b11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # M6 = (a10 - a00) * (b00 + b01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (a01 - a11) * (b10 + b11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Block 1: B cols 0,1 -> C cols 0,1
    for r_local in range(7):
        r_global = r_local
        u_terms, v_terms, w_terms = strassen_terms[r_local]
        
        for idx, val in u_terms:
            U[idx, r_global] += val
            
        for idx, val in v_terms:
            j = idx // 2
            k = idx % 2
            global_v_idx = j * p + k
            V[global_v_idx, r_global] += val
            
        for idx, val in w_terms:
            k = idx // 2
            i = idx % 2
            global_w_idx = k * n + i
            W[global_w_idx, r_global] += val

    # Block 2: B cols 2,3 -> C cols 2,3
    for r_local in range(7):
        r_global = 7 + r_local
        u_terms, v_terms, w_terms = strassen_terms[r_local]
        
        for idx, val in u_terms:
            U[idx, r_global] += val
            
        for idx, val in v_terms:
            j = idx // 2
            k_local = idx % 2
            k_global = k_local + 2
            global_v_idx = j * p + k_global
            V[global_v_idx, r_global] += val
            
        for idx, val in w_terms:
            k_local = idx // 2
            i = idx % 2
            k_global = k_local + 2
            global_w_idx = k_global * n + i
            W[global_w_idx, r_global] += val

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
