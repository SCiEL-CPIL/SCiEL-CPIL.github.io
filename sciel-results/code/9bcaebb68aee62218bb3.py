import time
import random
import numpy as np
from fractions import Fraction
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 28 Construction
    # Part 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    # Part 2: Schoolbook for j=2, k in {0..4}, i in {0,1}
    # Part 3a: Schoolbook for j in {0,1}, k=2, i in {0,1}
    # Part 3b: Strassen for i in {0,1}, j in {0,1}, k in {3,4}
    
    R = 28
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    r = 0
    
    # Helper to add a Strassen block
    def add_strassen(i_offset, j_offset, k_offset, start_r):
        # Local indices 0,1 map to global i_offset+li, j_offset+lj, k_offset+lk
        # Strassen terms for 2x2x2
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a01+a11)b00
        # m3 = a00(b10-b11)
        # m4 = a11(b00-b01)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # Coefficients for U (A part), V (B part), W (C part)
        # U[li, lj], V[lj, lk], W[lk, li]
        
        terms = [
            # m1
            {
                'U': {(0,0): 1, (1,1): 1},
                'V': {(0,0): 1, (1,1): 1},
                'W': {(0,0): 1, (1,1): 1}
            },
            # m2
            {
                'U': {(0,1): 1, (1,1): 1},
                'V': {(0,0): 1},
                'W': {(0,1): 1} # c01 gets m2
            },
            # m3
            {
                'U': {(0,0): 1},
                'V': {(1,0): 1, (1,1): -1},
                'W': {(0,1): 1} # c01 gets m3
            },
            # m4
            {
                'U': {(1,1): 1},
                'V': {(0,0): 1, (0,1): -1},
                'W': {(0,0): 1, (1,1): 1} # c00 gets m4, c11 gets m4
            },
            # m5
            {
                'U': {(0,0): 1, (0,1): 1},
                'V': {(1,1): 1},
                'W': {(1,0): 1, (0,0): -1} # c10 gets m5, c00 gets -m5
            },
            # m6
            {
                'U': {(1,0): 1, (0,0): -1},
                'V': {(0,0): 1, (0,1): 1},
                'W': {(1,0): 1, (1,1): 1} # c10 gets m6, c11 gets m6
            },
            # m7
            {
                'U': {(0,1): 1, (1,1): -1},
                'V': {(1,0): 1, (1,1): 1},
                'W': {(0,0): 1, (1,1): -1} # c00 gets m7, c11 gets -m7
            }
        ]
        
        for t in terms:
            for (li, lj), val in t['U'].items():
                u_idx = (i_offset + li) * m + (j_offset + lj)
                U[u_idx, start_r] += val
            for (lj, lk), val in t['V'].items():
                v_idx = (j_offset + lj) * p + (k_offset + lk)
                V[v_idx, start_r] += val
            for (lk, li), val in t['W'].items():
                w_idx = (k_offset + lk) * n + (i_offset + li)
                W[w_idx, start_r] += val
            start_r += 1
        return start_r

    # Part 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    r = add_strassen(0, 0, 0, r)
    
    # Part 2: Schoolbook for j=2, k in {0..4}, i in {0,1}
    for i in range(2):
        for k in range(5):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
            r += 1
            
    # Part 3a: Schoolbook for j in {0,1}, k=2, i in {0,1}
    for i in range(2):
        for j in range(2):
            u_idx = i * m + j
            v_idx = j * p + 2
            w_idx = 2 * n + i
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
            r += 1
            
    # Part 3b: Strassen for i in {0,1}, j in {0,1}, k in {3,4}
    r = add_strassen(0, 0, 3, r)
    
    # Verify and return
    # Check if r == 28
    if r != 28:
        # Fallback to schoolbook if construction failed
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, r] = 1.0
                    V[v_idx, r] = 1.0
                    W[w_idx, r] = 1.0
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
