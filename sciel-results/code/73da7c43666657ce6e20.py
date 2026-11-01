import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=7
    # Rank 26 for 3x3x3 part (k=0,1,2)
    # Rank 36 for 3x3x4 part (k=3,4,5,6)
    rank_part1 = 26
    rank_part2 = 3 * 3 * 4
    total_rank = rank_part1 + rank_part2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_term(col, u_indices, v_indices, w_indices):
        # u_indices: list of (idx, val)
        # v_indices: list of (idx, val)
        # w_indices: list of (idx, val)
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    col = 0
    
    # --- Part 1: 3x3x3 (k=0,1,2) ---
    # Indices:
    # A[i,j] -> i*3+j
    # B[j,k] -> j*7+k
    # C[k,i] -> k*3+i (Note: W index is k*n+i)
    
    # Strassen 2x2x2 for i,j in {0,1}, k in {0,1}
    # P1 = (a00+a11)(b00+b11)
    add_term(col, [(0, 1), (4, 1)], [(0, 1), (8, 1)], [(0, 1), (4, 1)])
    col += 1
    
    # P2 = (a10+a11)b00
    add_term(col, [(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, -1)])
    col += 1
    
    # P3 = a00(b10-b11)
    add_term(col, [(0, 1)], [(7, 1), (8, -1)], [(3, 1), (4, 1)])
    col += 1
    
    # P4 = a11(b01-b00)
    add_term(col, [(4, 1)], [(1, 1), (0, -1)], [(0, 1), (1, 1)])
    col += 1
    
    # P5 = (a00+a01)b11
    add_term(col, [(0, 1), (1, 1)], [(8, 1)], [(0, -1), (3, 1)])
    col += 1
    
    # P6 = (a10-a00)(b00+b01)
    add_term(col, [(3, 1), (0, -1)], [(0, 1), (1, 1)], [(4, 1)])
    col += 1
    
    # P7 = (a01-a11)(b10+b11)
    add_term(col, [(1, 1), (4, -1)], [(7, 1), (8, 1)], [(0, 1)])
    col += 1
    
    # Cross terms: a1*b2T (i in {0,1}, j=2, k in {0,1})
    # a02*b20 -> c00 (k=0, i=0 -> idx 0)
    add_term(col, [(2, 1)], [(14, 1)], [(0, 1)])
    col += 1
    
    # a02*b21 -> c01 (k=1, i=0 -> idx 3)
    add_term(col, [(2, 1)], [(15, 1)], [(3, 1)])
    col += 1
    
    # a12*b20 -> c10 (k=0, i=1 -> idx 1)
    add_term(col, [(5, 1)], [(14, 1)], [(1, 1)])
    col += 1
    
    # a12*b21 -> c11 (k=1, i=1 -> idx 4)
    add_term(col, [(5, 1)], [(15, 1)], [(4, 1)])
    col += 1
    
    # Cross terms: A22*b1 (i in {0,1}, j in {0,1}, k=2)
    # a00*b02 -> c02 (k=2, i=0 -> idx 6)
    add_term(col, [(0, 1)], [(2, 1)], [(6, 1)])
    col += 1
    
    # a01*b12 -> c02 (k=2, i=0 -> idx 6)
    add_term(col, [(1, 1)], [(9, 1)], [(6, 1)])
    col += 1
    
    # a10*b02 -> c12 (k=2, i=1 -> idx 7)
    add_term(col, [(3, 1)], [(2, 1)], [(7, 1)])
    col += 1
    
    # a11*b12 -> c12 (k=2, i=1 -> idx 7)
    add_term(col, [(4, 1)], [(9, 1)], [(7, 1)])
    col += 1
    
    # Cross terms: a2T*B22 (i=2, j in {0,1}, k in {0,1})
    # a20*b00 -> c20 (k=0, i=2 -> idx 2)
    add_term(col, [(6, 1)], [(0, 1)], [(2, 1)])
    col += 1
    
    # a21*b10 -> c20 (k=0, i=2 -> idx 2)
    add_term(col, [(7, 1)], [(7, 1)], [(2, 1)])
    col += 1
    
    # a20*b01 -> c21 (k=1, i=2 -> idx 5)
    add_term(col, [(6, 1)], [(1, 1)], [(5, 1)])
    col += 1
    
    # a21*b11 -> c21 (k=1, i=2 -> idx 5)
    add_term(col, [(7, 1)], [(8, 1)], [(5, 1)])
    col += 1
    
    # Cross terms: a1*b33 (i in {0,1}, j=2, k=2)
    # a02*b22 -> c02 (k=2, i=0 -> idx 6)
    add_term(col, [(2, 1)], [(16, 1)], [(6, 1)])
    col += 1
    
    # a12*b22 -> c12 (k=2, i=1 -> idx 7)
    add_term(col, [(5, 1)], [(16, 1)], [(7, 1)])
    col += 1
    
    # Cross terms: a33*b2T (i=2, j=2, k in {0,1})
    # a22*b20 -> c20 (k=0, i=2 -> idx 2)
    add_term(col, [(8, 1)], [(14, 1)], [(2, 1)])
    col += 1
    
    # a22*b21 -> c21 (k=1, i=2 -> idx 5)
    add_term(col, [(8, 1)], [(15, 1)], [(5, 1)])
    col += 1
    
    # Cross terms: a2T*b1 (i=2, j=2, k=2? No, j=2, k in {0,1} is a33*b2T)
    # a2T*b1 is i=2, j in {0,1}, k=2? No.
    # C22 = sum_j A2j Bj2.
    # j=0: A20 B02
    # j=1: A21 B12
    # j=2: A22 B22
    # We have A22 B22 (a33*b33) separate.
    # So we need A20 B02 and A21 B12.
    
    # a20*b02 -> c22 (k=2, i=2 -> idx 8)
    add_term(col, [(6, 1)], [(2, 1)], [(8, 1)])
    col += 1
    
    # a21*b12 -> c22 (k=2, i=2 -> idx 8)
    add_term(col, [(7, 1)], [(9, 1)], [(8, 1)])
    col += 1
    
    # Cross terms: a33*b33 (i=2, j=2, k=2)
    # a22*b22 -> c22 (k=2, i=2 -> idx 8)
    add_term(col, [(8, 1)], [(16, 1)], [(8, 1)])
    col += 1
    
    # --- Part 2: 3x3x4 (k=3,4,5,6) ---
    # Schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
