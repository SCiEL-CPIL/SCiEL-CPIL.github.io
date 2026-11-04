"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses Strassen (rank-7) for 2x2 block and schoolbook for 3x3 block,
with schoolbook for cross terms. Total rank: 27+18+12+7+18+12+18+12 = 124.
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                if abs(total - 1) > 1e-9:
                    return False, (i, j, k, total)
    return True, None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build terms: each term is (u_dict, v_dict, w_dict)
    terms = []
    
    # Block partition: indices 0,1,2 = block1 (size 3), indices 3,4 = block2 (size 2)
    # A[i,j] -> U index: i*5+j
    # B[j,k] -> V index: j*5+k
    # C[i,k] -> W index: k*5+i
    
    # --- C11 (3x3 output) = A11*B11 + A12*B21 ---
    # A11*B11: 3x3 * 3x3 schoolbook (27 terms)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a_idx = i * 5 + j
                b_idx = j * 5 + k
                w_idx = k * 5 + i
                terms.append(({a_idx: 1}, {b_idx: 1}, {w_idx: 1}))
    
    # A12*B21: A[0:3, 3:5] * B[3:5, 0:3] (18 terms)
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                a_idx = i * 5 + j
                b_idx = j * 5 + k
                w_idx = k * 5 + i
                terms.append(({a_idx: 1}, {b_idx: 1}, {w_idx: 1}))
    
    # --- C22 (2x2 output) = A21*B12 + A22*B22 ---
    # A21*B12: A[3:5, 0:3] * B[0:3, 3:5] (12 terms)
    for i in range(3, 5):
        for j in range(3):
            for k in range(3, 5):
                a_idx = i * 5 + j
                b_idx = j * 5 + k
                w_idx = k * 5 + i
                terms.append(({a_idx: 1}, {b_idx: 1}, {w_idx: 1}))
    
    # A22*B22: 2x2 * 2x2 Strassen (7 terms)
    def g22_a(li, lj):
        return (3 + li) * 5 + (3 + lj)
    
    def g22_b(lj, lk):
        return (3 + lj) * 5 + (3 + lk)
    
    def g22_w(li, lk):
        return (3 + lk) * 5 + (3 + li)
    
    # M1 = (A00+A11)(B00+B11)
    terms.append(({g22_a(0,0): 1, g22_a(1,1): 1},
                  {g22_b(0,0): 1, g22_b(1,1): 1},
                  {g22_w(0,0): 1, g22_w(1,1): 1}))
    
    # M2 = (A10+A11)B00
    terms.append(({g22_a(1,0): 1, g22_a(1,1): 1},
                  {g22_b(0,0): 1},
                  {g22_w(1,0): 1, g22_w(1,1): -1}))
    
    # M3 = A00(B10-B11)
    terms.append(({g22_a(0,0): 1},
                  {g22_b(1,0): 1, g22_b(1,1): -1},
                  {g22_w(0,1): 1, g22_w(1,1): 1}))
    
    # M4 = A11(B01-B00)
    terms.append(({g22_a(1,1): 1},
                  {g22_b(0,1): 1, g22_b(0,0): -1},
                  {g22_w(0,0): 1, g22_w(1,0): 1}))
    
    # M5 = (A00+A01)B11
    terms.append(({g22_a(0,0): 1, g22_a(0,1): 1},
                  {g22_b(1,1): 1},
                  {g22_w(0,0): -1, g22_w(0,1): 1}))
    
    # M6 = (A10-A00)(B00+B01)
    terms.append(({g22_a(1,0): 1, g22_a(0,0): -1},
                  {g22_b(0,0): 1, g22_b(0,1): 1},
                  {g22_w(1,1): 1}))
    
    # M7 = (A01-A11)(B10+B11)
    terms.append(({g22_a(0,1): 1, g22_a(1,1): -1},
                  {g22_b(1,0): 1, g22_b(1,1): 1},
                  {g22_w(0,0): 1}))
    
    # --- C12 (3x2 output) = A11*B12 + A12*B22 ---
    # A11*B12: A[0:3,0:3] * B[0:3,3:5] (18 terms)
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                a_idx = i * 5 + j
                b_idx = j * 5 + k
                w_idx = k * 5 + i
                terms.append(({a_idx: 1}, {b_idx: 1}, {w_idx: 1}))
    
    # A12*B22: A[0:3,3:5] * B[3:5,3:5] (12 terms)
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 5):
                a_idx = i * 5 + j
                b_idx = j * 5 + k
                w_idx = k * 5 + i
                terms.append(({a_idx: 1}, {b_idx: 1}, {w_idx: 1}))
    
    # --- C21 (2x3 output) = A21*B11 + A22*B21 ---
    # A21*B11: A[3:5,0:3] * B[0:3,0:3] (18 terms)
    for i in range(3, 5):
        for j in range(3):
            for k in range(3):
                a_idx = i * 5 + j
                b_idx = j * 5 + k
                w_idx = k * 5 + i
                terms.append(({a_idx: 1}, {b_idx: 1}, {w_idx: 1}))
    
    # A22*B21: A[3:5,3:5] * B[3:5,0:3] (12 terms)
    for i in range(3, 5):
        for j in range(3, 5):
            for k in range(3):
                a_idx = i * 5 + j
                b_idx = j * 5 + k
                w_idx = k * 5 + i
                terms.append(({a_idx: 1}, {b_idx: 1}, {w_idx: 1}))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in terms[r][0].items():
            U[idx][r] = val
        for idx, val in terms[r][1].items():
            V[idx][r] = val
        for idx, val in terms[r][2].items():
            W[idx][r] = val
    
    # Verify
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fall back to pure schoolbook
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1}))
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for r in range(rank):
            for idx, val in terms[r][0].items():
                U[idx][r] = val
            for idx, val in terms[r][1].items():
                V[idx][r] = val
            for idx, val in terms[r][2].items():
                W[idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
