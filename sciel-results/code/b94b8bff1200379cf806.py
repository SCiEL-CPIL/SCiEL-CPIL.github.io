"""
Matrix multiplication 4x4 * 4x7 tensor decomposition.
Uses Strassen's algorithm recursively to achieve rank 97 (49 + 48).
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    """
    Build a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns {'U': (16, rank), 'V': (28, rank), 'W': (28, rank)}
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Strategy: Split 4x4x7 into 4x4x4 + 4x4x3
    # 4x4x4 via recursive Strassen: rank 49
    # 4x4x3 via schoolbook: rank 48
    # Total rank: 97
    
    # Build 4x4x4 Strassen decomposition
    U_444, V_444, W_444 = _strassen_4x4x4()
    rank_444 = len(U_444[0])  # 49
    
    # Build 4x4x3 schoolbook decomposition
    U_443, V_443, W_443 = _schoolbook_4x4x3()
    rank_443 = len(U_443[0])  # 48
    
    total_rank = rank_444 + rank_443  # 97
    
    # Initialize full matrices
    # U: (n*m, rank) = (16, 97)
    # V: (m*p, rank) = (28, 97)  
    # W: (p*n, rank) = (28, 97)
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]
    W = [[0] * total_rank for _ in range(28)]
    
    # Fill in 4x4x4 part (columns 0 to rank_444-1)
    # U_444[i*4+j, c] -> U[i*4+j, c]
    # V_444[j*4+k, c] -> V[j*7+k, c] for k=0..3 (first 4 columns of B)
    # W_444[k*4+i, c] -> W[k*4+i, c] for k=0..3 (first 4 rows of C)
    for c in range(rank_444):
        for idx in range(16):
            U[idx][c] = U_444[idx][c]
        for j in range(4):
            for k in range(4):
                V[j * 7 + k][c] = V_444[j * 4 + k][c]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][c] = W_444[k * 4 + i][c]
    
    # Fill in 4x4x3 part (columns rank_444 to total_rank-1)
    # U_443[i*4+j, c] -> U[i*4+j, rank_444 + c]
    # V_443[j*3+k, c] -> V[j*7+(4+k), rank_444 + c] for k=0..2 (columns 4,5,6 of B)
    # W_443[k*4+i, c] -> W[(4+k)*4+i, rank_444 + c] for k=0..2 (rows 4,5,6 of C)
    for c in range(rank_443):
        col = rank_444 + c
        for idx in range(16):
            U[idx][col] = U_443[idx][c]
        for j in range(4):
            for k in range(3):
                V[j * 7 + (4 + k)][col] = V_443[j * 3 + k][c]
        for k in range(3):
            for i in range(4):
                W[(4 + k) * 4 + i][col] = W_443[k * 4 + i][c]
    
    return {"U": U, "V": V, "W": W}


def _strassen_4x4x4():
    """
    Build rank-49 Strassen decomposition for 4x4x4 matrix multiplication.
    Uses recursive Strassen: 4x4x4 = 7 * (2x2x2), each 2x2x2 has rank 7.
    Returns U (16x49), V (16x49), W (16x49).
    """
    # 2x2x2 Strassen base case
    u2, v2, w2 = _strassen_2x2x2()
    
    # 4x4x4: Split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A*B computed via Strassen
    
    # 7 top-level terms, each is a 2x2x2 product
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    # For each top-level term t (0..6) and each sub-term s (0..6):
    # Column index = t * 7 + s
    
    # Define which A and B blocks are used for each top-level term
    # And their signs
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Each Pi involves specific A blocks and B blocks
    # For term t, the A side uses blocks a1, a2 (with signs) and B side uses blocks b1, b2 (with signs)
    
    # Term definitions: (a_blocks, b_blocks) where each is list of (block_idx, sign)
    # Block indices: 00, 01, 10, 11 -> (row_block, col_block)
    terms = [
        # t=0: P1 = (A00+A11)(B00+B11)
        {
            'a_blocks': [((0,0), 1), ((1,1), 1)],
            'b_blocks': [((0,0), 1), ((1,1), 1)],
            'c_blocks': [((0,0), 1), ((0,1), -1), ((1,0), -1), ((1,1), 1)]
        },
        # t=1: P2 = (A10+A11)B00
        {
            'a_blocks': [((1,0), 1), ((1,1), 1)],
            'b_blocks': [((0,0), 1)],
            'c_blocks': [((1,0), 1), ((1,1), -1)]
        },
        # t=2: P3 = A00(B10-B11)
        {
            'a_blocks': [((0,0), 1)],
            'b_blocks': [((1,0), 1), ((1,1), -1)],
            'c_blocks': [((0,1), 1), ((1,1), 1)]
        },
        # t=3: P4 = A11(B01-B10)
        {
            'a_blocks': [((1,1), 1)],
            'b_blocks': [((0,1), 1), ((1,0), -1)],
            'c_blocks': [((0,0), 1), ((1,0), 1)]
        },
        # t=4: P5 = (A00+A01)B11
        {
            'a_blocks': [((0,0), 1), ((0,1), 1)],
            'b_blocks': [((1,1), 1)],
            'c_blocks': [((0,0), -1), ((1,0), 1)]
        },
        # t=5: P6 = (A10-A00)(B00+B01)
        {
            'a_blocks': [((1,0), 1), ((0,0), -1)],
            'b_blocks': [((0,0), 1), ((0,1), 1)],
            'c_blocks': [((1,1), 1)]
        },
        # t=6: P7 = (A01-A11)(B10+B11)
        {
            'a_blocks': [((0,1), 1), ((1,1), -1)],
            'b_blocks': [((1,0), 1), ((1,1), 1)],
            'c_blocks': [((0,0), 1), ((0,1), 1)]
        },
    ]
    
    # For each top-level term, apply the 2x2x2 Strassen decomposition
    for t, term in enumerate(terms):
        for s in range(7):
            col = t * 7 + s
            
            # Apply 2x2x2 Strassen coefficients for sub-term s
            # u2[i*2+j, s], v2[j*2+k, s], w2[k*2+i, s]
            
            # For U: combine A block contributions
            for (ai, aj), sign_a in term['a_blocks']:
                # A block (ai, aj) occupies rows ai*2..ai*2+1, cols aj*2..aj*2+1
                # In U: row index = i*4 + j where i is row block, j is col block
                # For the 2x2x2 sub-problem, local indices (li, lj) map to global (ai*2+li, aj*2+lj)
                for li in range(2):
                    for lj in range(2):
                        u_val = u2[li * 2 + lj][s]
                        if u_val != 0:
                            global_i = ai * 2 + li
                            global_j = aj * 2 + lj
                            U[global_i * 4 + global_j][col] += sign_a * u_val
            
            # For V: combine B block contributions
            for (bi, bj), sign_b in term['b_blocks']:
                # B block (bi, bj) occupies rows bi*2..bi*2+1, cols bj*2..bj*2+1
                # In V: row index = j*4 + k where j is row index in B, k is col index
                # For the 2x2x2 sub-problem, local indices (lj, lk) map to global (bi*2+lj, bj*2+lk)
                for lj in range(2):
                    for lk in range(2):
                        v_val = v2[lj * 2 + lk][s]
                        if v_val != 0:
                            global_j = bi * 2 + lj
                            global_k = bj * 2 + lk
                            V[global_j * 4 + global_k][col] += sign_b * v_val
            
            # For W: combine C block contributions
            for (ci, ck), sign_c in term['c_blocks']:
                # C block (ci, ck) occupies rows ck*2..ck*2+1, cols ci*2..ci*2+1
                # In W: row index = k*4 + i where k is col index in C, i is row index
                # For the 2x2x2 sub-problem, local indices (lk, li) map to global (ck*2+lk, ci*2+li)
                for lk in range(2):
                    for li in range(2):
                        w_val = w2[lk * 2 + li][s]
                        if w_val != 0:
                            global_k = ck * 2 + lk
                            global_i = ci * 2 + li
                            W[global_k * 4 + global_i][col] += sign_c * w_val
    
    return U, V, W


def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 Strassen decomposition.
    Each is 4x7.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    u = [[0]*7 for _ in range(4)]
    v = [[0]*7 for _ in range(4)]
    w = [[0]*7 for _ in range(4)]
    
    # c=0: M1 = (A00+A11)(B00+B11)
    # C00 += M1, C01 -= M1, C10 -= M1, C11 += M1
    u[0][0] = 1; u[3][0] = 1  # A00, A11
    v[0][0] = 1; v[3][0] = 1  # B00, B11
    w[0][0] = 1; w[1][0] = -1; w[2][0] = -1; w[3][0] = 1  # C00, C01, C10, C11
    
    # c=1: M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2
    u[2][1] = 1; u[3][1] = 1  # A10, A11
    v[0][1] = 1  # B00
    w[2][1] = 1; w[3][1] = -1  # C10, C11
    
    # c=2: M3 = A00(B10-B11)
    # C01 += M3, C11 += M3
    u[0][2] = 1  # A00
    v[2][2] = 1; v[3][2] = -1  # B10, B11
    w[1][2] = 1; w[3][2] = 1  # C01, C11
    
    # c=3: M4 = A11(B01-B10)
    # C00 += M4, C10 += M4
    u[3][3] = 1  # A11
    v[1][3] = 1; v[2][3] = -1  # B01, B10
    w[0][3] = 1; w[2][3] = 1  # C00, C10
    
    # c=4: M5 = (A00+A01)B11
    # C00 -= M5, C10 += M5
    u[0][4] = 1; u[1][4] = 1  # A00, A01
    v[3][4] = 1  # B11
    w[0][4] = -1; w[2][4] = 1  # C00, C10
    
    # c=5: M6 = (A10-A00)(B00+B01)
    # C11 += M6
    u[2][5] = 1; u[0][5] = -1  # A10, A00
    v[0][5] = 1; v[1][5] = 1  # B00, B01
    w[3][5] = 1  # C11
    
    # c=6: M7 = (A01-A11)(B10+B11)
    # C00 += M7, C01 += M7
    u[1][6] = 1; u[3][6] = -1  # A01, A11
    v[2][6] = 1; v[3][6] = 1  # B10, B11
    w[0][6] = 1; w[1][6] = 1  # C00, C01
    
    return u, v, w


def _schoolbook_4x4x3():
    """
    Build rank-48 schoolbook decomposition for 4x4x3 matrix multiplication.
    Returns U (16x48), V (12x48), W (12x48).
    """
    rank = 48
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4+j][col] = 1
                V[j*3+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    return U, V, W

# EVOLVE-BLOCK-END
