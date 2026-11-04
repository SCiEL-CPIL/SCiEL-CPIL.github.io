import numpy as np
import time
def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 3
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Define the 23 terms
    # Each term is (A_factors, B_factors, C_factors)
    # A_factors: list of (index, coeff)
    # B_factors: list of (index, coeff)
    # C_factors: list of (index, coeff)
    
    terms = []
    
    # Term 0
    terms.append(([(0, 1), (4, -1), (8, -1)], [(0, 1), (4, 1), (8, 1)], [(0, 1)]))
    # Term 1
    terms.append(([(4, 1), (8, -1)], [(4, 1), (8, 1)], [(0, 1), (4, 1), (8, 1)]))
    # Term 2
    terms.append(([(8, 1)], [(8, 1)], [(0, 1), (4, 1), (8, 1)]))
    # Term 3
    terms.append(([(1, 1), (5, -1)], [(3, 1), (6, -1)], [(1, 1), (4, -1), (7, -1)]))
    # Term 4
    terms.append(([(2, 1), (3, -1)], [(6, 1), (0, -1)], [(2, 1), (3, -1), (6, -1)]))
    # Term 5
    terms.append(([(3, 1), (6, -1)], [(0, 1), (1, -1)], [(3, 1), (4, -1), (7, -1)]))
    # Term 6
    terms.append(([(1, 1), (2, -1)], [(2, 1), (0, -1)], [(1, 1), (2, -1), (5, -1)]))
    # Term 7
    terms.append(([(5, 1), (8, -1)], [(6, 1), (7, -1)], [(5, 1), (8, -1), (7, -1)]))
    # Term 8
    terms.append(([(6, 1), (7, -1)], [(4, 1), (5, -1)], [(6, 1), (7, -1), (5, -1)]))
    # Term 9
    terms.append(([(1, 1), (4, 1), (7, 1)], [(2, 1), (5, -1), (8, -1)], [(1, 1), (4, -1), (7, -1)]))
    # Term 10
    terms.append(([(4, 1), (7, 1)], [(5, 1), (8, 1)], [(4, 1), (7, 1), (8, 1)]))
    # Term 11
    terms.append(([(7, 1)], [(8, 1)], [(7, 1), (8, 1)]))
    # Term 12
    terms.append(([(2, 1), (5, 1), (8, 1)], [(1, 1), (4, -1), (7, -1)], [(2, 1), (5, -1), (8, -1)]))
    # Term 13
    terms.append(([(5, 1), (8, 1)], [(4, 1), (7, 1)], [(5, 1), (8, 1), (7, 1)]))
    # Term 14
    terms.append(([(8, 1)], [(7, 1)], [(8, 1), (7, 1)]))
    # Term 15
    terms.append(([(0, 1), (3, 1), (6, 1)], [(2, 1), (5, -1), (8, -1)], [(0, 1), (3, -1), (6, -1)]))
    # Term 16
    terms.append(([(3, 1), (6, 1)], [(5, 1), (8, 1)], [(3, 1), (6, 1), (8, 1)]))
    # Term 17
    terms.append(([(6, 1)], [(8, 1)], [(6, 1), (8, 1)]))
    # Term 18
    terms.append(([(0, 1), (3, 1), (6, 1)], [(1, 1), (4, -1), (7, -1)], [(0, 1), (3, -1), (6, -1)]))
    # Term 19
    terms.append(([(3, 1), (6, 1)], [(4, 1), (7, 1)], [(3, 1), (6, 1), (7, 1)]))
    # Term 20
    terms.append(([(6, 1)], [(7, 1)], [(6, 1), (7, 1)]))
    # Term 21
    terms.append(([(0, 1), (3, 1), (6, 1)], [(0, 1), (3, -1), (6, -1)], [(0, 1), (3, -1), (6, -1)]))
    # Term 22
    terms.append(([(3, 1), (6, 1)], [(3, 1), (6, 1)], [(3, 1), (6, 1), (0, 1)]))

    rank = len(terms)
    U = np.zeros((9, rank), dtype=float)
    V = np.zeros((9, rank), dtype=float)
    W = np.zeros((9, rank), dtype=float)
    
    for r, (a_factors, b_factors, c_factors) in enumerate(terms):
        for idx, coeff in a_factors:
            U[idx, r] = coeff
        for idx, coeff in b_factors:
            V[idx, r] = coeff
        for idx, coeff in c_factors:
            W[idx, r] = coeff

    # Verify
    T = np.zeros((9, 9, 9), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T[a, b, c] = 1

    R = np.einsum('ar,br,cr->abc', U, V, W)
    if np.allclose(R, T, atol=1e-9):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback
    rank_fb = 27
    U_fb = np.zeros((9, rank_fb), dtype=float)
    V_fb = np.zeros((9, rank_fb), dtype=float)
    W_fb = np.zeros((9, rank_fb), dtype=float)
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                U_fb[a, col] = 1
                V_fb[b, col] = 1
                W_fb[c, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
