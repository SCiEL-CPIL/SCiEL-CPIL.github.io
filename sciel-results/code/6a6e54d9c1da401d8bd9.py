import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _build_strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # T1: (A11+A22)(B11+B22) -> C11, C22
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # T2: (A21+A22)B11 -> C21, C22
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # T3: A11(B12-B22) -> C12, C22
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # T4: A22(B21-B11) -> C11, C21
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # T5: (A11+A12)B22 -> C11, C12
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # T6: (A21-A11)(B11+B12) -> C22
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # T7: (A12-A22)(B21+B22) -> C11
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_223():
    """
    Builds 2x2x3 decomposition.
    Rank 11.
    """
    U1, V1, W1 = _build_strassen_222()
    
    U2 = [[0]*4 for _ in range(4)]
    V2 = [[0]*4 for _ in range(2)]
    W2 = [[0]*4 for _ in range(2)]
    
    # Schoolbook for 2x2 x 2x1 -> 2x1
    U2[0][0] = 1; V2[0][0] = 1; W2[0][0] = 1
    U2[1][1] = 1; V2[1][1] = 1; W2[0][1] = 1
    U2[2][2] = 1; V2[0][2] = 1; W2[1][2] = 1
    U2[3][3] = 1; V2[1][3] = 1; W2[1][3] = 1
    
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(6)]
    
    # Strassen for first 2 columns
    v_map = [0, 1, 3, 4]
    w_map = [0, 1, 3, 4]
    for a in range(4):
        for r in range(7):
            U[a][r] = U1[a][r]
    for b in range(4):
        for r in range(7):
            V[v_map[b]][r] = V1[b][r]
    for c in range(4):
        for r in range(7):
            W[w_map[c]][r] = W1[c][r]
    
    # Schoolbook for last column
    v_map2 = [2, 5]
    w_map2 = [2, 5]
    for a in range(4):
        for r in range(4):
            U[a][r+7] = U2[a][r]
    for b in range(2):
        for r in range(4):
            V[v_map2[b]][r+7] = V2[b][r]
    for c in range(2):
        for r in range(4):
            W[w_map2[c]][r+7] = W2[c][r]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115
    
    U_dim = n * m  # 12
    V_dim = m * p  # 24
    W_dim = p * n  # 18
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    def make_vec(dim, indices_vals):
        vec = [0] * dim
        for idx, val in indices_vals:
            vec[idx] = val
        return vec

    U223, V223, W223 = _build_223()
    
    # Part 1: Row 2 of A (1x4) times B (4x6) -> Row 2 of C (1x6)
    # Schoolbook: 24 terms
    for j in range(4):
        for k in range(6):
            u = make_vec(U_dim, [(8+j, 1)])
            v = make_vec(V_dim, [(j*6+k, 1)])
            w = make_vec(W_dim, [(k*3+2, 1)])
            add_term(u, v, w)
            
    # Part 2: Rows 0-1 of A (2x4) times B (4x6) -> Rows 0-1 of C (2x6)
    # Split A into two 2x2 blocks: A[:,0:2] and A[:,2:4]
    # For each 2x2 block, multiply by all of B (4x6)
    # 2x2 x 4x6 -> 2x6
    # Split B into four 2x3 blocks
    # 2x2 x 2x3 -> 2x3 has rank 11
    # So for each 2x2 A block and each 2x3 B block, we get 11 terms
    # 2 A blocks * 4 B blocks * 11 = 88 terms
    
    for j_block in range(2):  # Two 2x2 blocks in A
        j_offset = j_block * 2
        for b_row_block in range(2):  # Two 2x3 blocks in B by row
            b_row_offset = b_row_block * 2
            for k_block in range(2):  # Two 2x3 blocks in B by column
                k_offset = k_block * 3
                
                for r in range(11):
                    u_vec = make_vec(U_dim, [])
                    v_vec = make_vec(V_dim, [])
                    w_vec = make_vec(W_dim, [])
                    
                    # U: 2x2 A block at cols j_offset, j_offset+1
                    u_map = [
                        0*4 + j_offset,
                        0*4 + j_offset + 1,
                        1*4 + j_offset,
                        1*4 + j_offset + 1
                    ]
                    for a in range(4):
                        if U223[a][r] != 0:
                            u_vec[u_map[a]] = U223[a][r]
                            
                    # V: 2x3 B block at rows b_row_offset..b_row_offset+1, cols k_offset..k_offset+2
                    v_map = [
                        (b_row_offset+0)*6 + k_offset + 0,
                        (b_row_offset+0)*6 + k_offset + 1,
                        (b_row_offset+0)*6 + k_offset + 2,
                        (b_row_offset+1)*6 + k_offset + 0,
                        (b_row_offset+1)*6 + k_offset + 1,
                        (b_row_offset+1)*6 + k_offset + 2
                    ]
                    for b in range(6):
                        if V223[b][r] != 0:
                            v_vec[v_map[b]] = V223[b][r]
                            
                    # W: 2x3 C block at rows 0-1, cols k_offset..k_offset+2
                    w_map = [
                        (k_offset+0)*3 + 0,
                        (k_offset+1)*3 + 0,
                        (k_offset+2)*3 + 0,
                        (k_offset+0)*3 + 1,
                        (k_offset+1)*3 + 1,
                        (k_offset+2)*3 + 1
                    ]
                    for c in range(6):
                        if W223[c][r] != 0:
                            w_vec[w_map[c]] = W223[c][r]
                            
                    add_term(u_vec, v_vec, w_vec)
                    
    rank = len(terms)
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
    for r, (u, v, w) in enumerate(terms):
        for a in range(U_dim):
            if u[a] != 0:
                U[a][r] = u[a]
        for b in range(V_dim):
            if v[b] != 0:
                V[b][r] = v[b]
        for c in range(W_dim):
            if w[c] != 0:
                W[c][r] = w[c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
