import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0

def _winograd_2x2x2():
    """
    Returns U, V, W matrices (4x7) for 2x2x2 matrix multiplication.
    """
    U2 = [
        [1, 0, 0, 1, 0, 1, 0],  # a
        [0, 0, 0, 1, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 1, 0, 0, 0, 1],  # d
    ]
    V2 = [
        [1, 1, -1, 0, 0, 1, 0],   # e
        [0, 0, 1, 0, 0, 0, 1],    # f
        [0, 0, 0, 0, 1, 1, 0],    # g
        [1, 0, 0, 1, -1, 0, 1],   # h
    ]
    W2 = [
        [1, -1, 0, 0, -1, 1, 0],   # C[0,0]
        [0, 0, -1, 1, 1, 0, 1],    # C[0,1]
        [0, 1, 0, -1, 0, 1, -1],   # C[1,0]
        [1, 0, -1, 0, 1, -1, 0],   # C[1,1]
    ]
    return U2, V2, W2

def _strassen_4x4x4():
    """
    Constructs U, V, W for 4x4x4 matrix multiplication using Strassen's algorithm.
    Rank = 49.
    """
    U2, V2, W2 = _winograd_2x2x2()
    
    # Strassen terms for 2x2 blocks of 2x2 matrices
    # Each term k is defined by coefficients for A blocks (X_k) and B blocks (Y_k)
    # A blocks: 00, 01, 10, 11 (indices 0,1,2,3)
    # B blocks: 00, 01, 10, 11 (indices 0,1,2,3)
    
    # X_k (coeffs for A blocks) and Y_k (coeffs for B blocks)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    strassen_terms = [
        # (coeffs_A, coeffs_B)
        # coeffs is dict {block_idx: coeff}
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    # Global U, V, W for 4x4x4
    # U: 16 rows (A elements), 49 cols
    # V: 16 rows (B elements), 49 cols
    # W: 16 rows (C elements), 49 cols
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    col_offset = 0
    for k in range(7):
        coeffs_A = strassen_terms[k][0]
        coeffs_B = strassen_terms[k][1]
        
        # For each of the 7 Winograd terms
        for w in range(7):
            c = col_offset + w
            
            # Update U
            # U rows are A[i,j]. Block idx b corresponds to A[b//2, b%2].
            # Within block, U2[row] corresponds to element.
            for b_idx, coeff_a in coeffs_A.items():
                # Block b_idx is A[b_idx//2, b_idx%2]
                # Elements in this block:
                # A[bi//2, bi%2], A[bi//2, bi%2+1], A[bi//2+1, bi%2], A[bi//2+1, bi%2+1]
                # Indices in global U:
                # i = bi//2, j = bi%2
                # u0 = i*4 + j
                # u1 = i*4 + j + 1
                # u2 = (i+1)*4 + j
                # u3 = (i+1)*4 + j + 1
                i_base = b_idx // 2
                j_base = b_idx % 2
                u_indices = [
                    i_base * 4 + j_base,
                    i_base * 4 + j_base + 1,
                    (i_base + 1) * 4 + j_base,
                    (i_base + 1) * 4 + j_base + 1
                ]
                for r in range(4):
                    U[u_indices[r]][c] += coeff_a * U2[r][w]
            
            # Update V
            for b_idx, coeff_b in coeffs_B.items():
                i_base = b_idx // 2
                j_base = b_idx % 2
                v_indices = [
                    i_base * 4 + j_base,
                    i_base * 4 + j_base + 1,
                    (i_base + 1) * 4 + j_base,
                    (i_base + 1) * 4 + j_base + 1
                ]
                for r in range(4):
                    V[v_indices[r]][c] += coeff_b * V2[r][w]
            
            # Update W
            # Output C is sum of M_k contributions.
            # C = C00 M1 + ... 
            # C00 = M1 - M4 - M6 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 - M5 + M6
            # We need to map the Winograd output (2x2) to the global C (4x4) blocks.
            # The Winograd W2 gives the coefficients for the 2x2 output of X_k Y_k.
            # But X_k Y_k is just the k-th Strassen product M_k (a 2x2 matrix).
            # So the 2x2 output of Winograd corresponds to the 2x2 matrix M_k.
            # M_k contributes to C blocks with specific signs.
            
            # Let's define the contribution of M_k to C blocks.
            # C00: M1 - M4 - M6 + M7
            # C01: M3 + M5
            # C10: M2 + M4
            # C11: M1 - M3 - M5 + M6
            
            # For current k, what is the coefficient for each C block?
            coeff_C = {0: 0, 1: 0, 2: 0, 3: 0}
            if k == 0: coeff_C = {0: 1, 1: 0, 2: 0, 3: 1}
            elif k == 1: coeff_C = {0: 0, 1: 0, 2: 1, 3: 0}
            elif k == 2: coeff_C = {0: 0, 1: 1, 2: 0, 3: -1}
            elif k == 3: coeff_C = {0: -1, 1: 0, 2: 1, 3: 0}
            elif k == 4: coeff_C = {0: 0, 1: 1, 2: 0, 3: -1}
            elif k == 5: coeff_C = {0: -1, 1: 0, 2: 0, 3: 1}
            elif k == 6: coeff_C = {0: 1, 1: 0, 2: 0, 3: -1}
            
            for c_block_idx, coeff_c in coeff_C.items():
                if coeff_c == 0: continue
                i_base = c_block_idx // 2
                j_base = c_block_idx % 2
                w_indices = [
                    i_base * 4 + j_base,
                    i_base * 4 + j_base + 1,
                    (i_base + 1) * 4 + j_base,
                    (i_base + 1) * 4 + j_base + 1
                ]
                for r in range(4):
                    W[w_indices[r]][c] += coeff_c * W2[r][w]
            
            col_offset += 1
            
    return U, V, W

def _matvec_4x4x1():
    """
    Constructs U, V, W for 4x4 matrix * 4x1 vector.
    Rank = 16.
    Returns U (16x16), V (4x16), W (4x16).
    Note: V and W shapes here are local to the sub-problem.
    We need to map them to global indices later.
    """
    U = [[0] * 16 for _ in range(16)]
    V = [[0] * 16 for _ in range(4)]
    W = [[0] * 16 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U[i * 4 + j][c] = 1
            V[j][c] = 1
            W[i][c] = 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # Total Rank = 49 (for B cols 0-3) + 49 (for B cols 4-7) + 16 (for B col 8) = 114
    total_rank = 114
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: B cols 0-3 (4x4)
    U1, V1, W1 = _strassen_4x4x4()
    for r in range(16):
        for c in range(49):
            U[r][c] = U1[r][c]
            # V1 rows are B[0:4, 0:4]. Global V rows are B[j, k].
            # V1[r] corresponds to B[r//4, r%4].
            # Global V index = j*9 + k.
            j = r // 4
            k = r % 4
            V[j * 9 + k][c] = V1[r][c]
            # W1 rows are C[0:4, 0:4]. Global W rows are C[i, k].
            # W1[r] corresponds to C[r//4, r%4].
            # Global W index = k*4 + i.
            i = r // 4
            k_out = r % 4
            W[k_out * 4 + i][c] = W1[r][c]
            
    # Part 2: B cols 4-7 (4x4)
    # Same Strassen decomposition, but B indices shifted.
    for r in range(16):
        for c in range(49):
            U[r][49 + c] = U1[r][c]
            j = r // 4
            k = r % 4
            # B col k+4
            V[j * 9 + (k + 4)][49 + c] = V1[r][c]
            i = r // 4
            k_out = r % 4
            # C col k_out+4
            W[(k_out + 4) * 4 + i][49 + c] = W1[r][c]
            
    # Part 3: B col 8 (4x1)
    U3, V3, W3 = _matvec_4x4x1()
    for r in range(16):
        for c in range(16):
            U[r][98 + c] = U3[r][c]
            # V3 rows are B[0:4, 0]. Global V row is B[j, 8].
            j = r
            V[j * 9 + 8][98 + c] = V3[r][c]
            # W3 rows are C[0:4, 0]. Global W row is C[i, 8].
            i = r
            W[8 * 4 + i][98 + c] = W3[r][c]
            
    return {"U": U, "V": V, "W": W}
