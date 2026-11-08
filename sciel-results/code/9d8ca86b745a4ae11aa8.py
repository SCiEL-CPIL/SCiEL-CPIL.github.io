import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_term(u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank overflow")
        for i, c in zip(u_idx, u_coeff):
            U[i, col] += c
        for i, c in zip(v_idx, v_coeff):
            V[i, col] += c
        for i, c in zip(w_idx, w_coeff):
                W[i, col] += c
        col += 1

    def add_strassen_block(k_start):
        a00, a01, a10, a11 = 0, 1, 3, 4
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
        add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
        add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
        add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
        add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
        add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
        add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    b06 = 6
    b16 = 7 + 6
    c06 = 6 * 2
    c16 = 6 * 2 + 1
    
    add_term([0], [1], [b06], [1], [c06], [1])
    add_term([1], [1], [b16], [1], [c06], [1])
    add_term([3], [1], [b06], [1], [c16], [1])
    add_term([4], [1], [b16], [1], [c16], [1])
    
    for k in range(7):
        b2k = 14 + k
        c0k = k * 2
        c1k = k * 2 + 1
        add_term([2], [1], [b2k], [1], [c0k], [1])
        add_term([5], [1], [b2k], [1], [c1k], [1])
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
