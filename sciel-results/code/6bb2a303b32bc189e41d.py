"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split p into p1=4 and p2=5
    p1 = 4
    p2 = 5
    
    # Rank for 4x4x4 part using Strassen (7*7=49)
    rank1 = 49
    # Rank for 4x4x5 part using Schoolbook (4*4*5=80)
    rank2 = m * p2 * n # 4*5*4 = 80
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 using Strassen
    # We need to generate the 49 columns
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Strassen factors for 2x2x2
    # U22: 4x7, V22: 4x7, W22: 4x7
    # Col 0 (M1): U: 1,0,0,1; V: 1,0,0,1; W: 1,0,0,-1 (C00 += M1, C11 -= M1? No, C00=M1+M4-M5+M7, C11=M1-M2+M3+M6)
    # Let's define the contributions to C.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W22 rows correspond to C indices 0,1,2,3.
    # W22[0] (C00): [1, 0, 0, 1, -1, 0, 1]
    # W22[1] (C01): [0, 0, 1, 0, 1, 0, 0]
    # W22[2] (C10): [0, 1, 0, 1, 0, 0, 0]
    # W22[3] (C11): [1, -1, 1, 0, 0, 1, 0]
    
    # U22 rows correspond to A indices 0,1,2,3.
    # U22[0] (A00): [1, 0, 1, 0, 1, -1, 0]
    # U22[1] (A01): [0, 0, 0, 0, 1, 0, 1]
    # U22[2] (A10): [0, 1, 0, 0, 0, 1, 0]
    # U22[3] (A11): [1, 1, 0, 1, 0, 0, -1]
    
    # V22 rows correspond to B indices 0,1,2,3.
    # V22[0] (B00): [1, 1, 0, -1, 0, 1, 0]
    # V22[1] (B01): [0, 0, 1, 0, 0, 1, 0]
    # V22[2] (B10): [0, 0, -1, 1, 0, 0, 1]
    # V22[3] (B11): [1, 0, -1, 0, 1, 0, 1]
    
    U22 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V22 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, -1, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W22 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Block indices for 4x4 matrix:
    # A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # Each is 2x2.
    # Global A index for block (bi, bj) and local (li, lj) is (bi*2+li)*2 + (bj*2+lj)
    
    # For each of the 7 block multiplications M_k:
    # M1: (A00+A11)(B00+B11) -> X = A00+A11, Y = B00+B11
    # M2: (A10+A11)B00 -> X = A10+A11, Y = B00
    # M3: A00(B01-B11) -> X = A00, Y = B01-B11
    # M4: A11(B10-B00) -> X = A11, Y = B10-B00
    # M5: (A00+A01)B11 -> X = A00+A01, Y = B11
    # M6: (A10-A00)(B00+B01) -> X = A10-A00, Y = B00+B01
    # M7: (A01-A11)(B10+B11) -> X = A01-A11, Y = B10+B11
    
    # We need to map X and Y to the global U and V vectors.
    # For a given M_k, X is a 2x2 matrix formed by linear combination of A blocks.
    # Y is a 2x2 matrix formed by linear combination of B blocks.
    # Then we use U22, V22, W22 to decompose X*Y.
    
    # Let's define the block combinations
    # X_combinations: list of 7, each is a dict {block_idx: coeff}
    # Y_combinations: list of 7, each is a dict {block_idx: coeff}
    
    # Block indices: 0:00, 1:01, 2:10, 3:11
    
    X_combos = [
        {0: 1, 3: 1}, # M1
        {2: 1, 3: 1}, # M2
        {0: 1},       # M3
        {3: 1},       # M4
        {0: 1, 1: 1}, # M5
        {2: 1, 0: -1},# M6
        {1: 1, 3: -1} # M7
    ]
    
    Y_combos = [
        {0: 1, 3: 1}, # M1
        {0: 1},       # M2
        {1: 1, 3: -1},# M3
        {2: 1, 0: -1},# M4
        {3: 1},       # M5
        {0: 1, 1: 1}, # M6
        {2: 1, 3: 1}  # M7
    ]
    
    # C combinations for the 4x4 result
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_combos = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00 (M1, M4, M5, M7)
        {2: 1, 4: 1},              # C01 (M3, M5)
        {1: 1, 3: 1},              # C10 (M2, M4)
        {0: 1, 1: -1, 2: 1, 5: 1} # C11 (M1, M2, M3, M6)
    ]
    
    col = 0
    for k in range(7):
        # For each of the 7 Strassen steps in this block multiplication
        for s in range(7):
            # U contribution
            # X is formed by X_combos[k]
            # Local U22 column s gives coefficients for local A indices
            # We need to map local A indices to global A indices
            for li in range(4): # 0,1,2,3 local A indices
                coeff_u = U22[li, s]
                if coeff_u != 0:
                    # li corresponds to local (li//2, li%2)
                    # We need to find which global block this local index belongs to
                    # But X is a sum of blocks.
                    # The local index li applies to ALL blocks in X_combos[k] with the same local offset?
                    # No, X is a 2x2 matrix. The entries of X are linear combinations of A entries.
                    # X[li//2, li%2] = sum_{block in X_combos[k]} coeff * A_block[li//2, li%2]
                    # So for a fixed local index li, the global coefficient for A_global is the sum of coeffs from all blocks in X_combos[k] that contain this local index.
                    # Since all blocks are 2x2, they all have the same local indices.
                    # So for a fixed li, the contribution to U is:
                    # For each block b in X_combos[k]:
                    #   global_row = b*4 + li
                    #   U[global_row][col] += coeff_u * X_combos[k][b]
                    
                    for b, coeff_b in X_combos[k].items():
                        global_row = b * 4 + li
                        U[global_row][col] += coeff_u * coeff_b
            
            # V contribution
            for lj in range(4):
                coeff_v = V22[lj, s]
                if coeff_v != 0:
                    for b, coeff_b in Y_combos[k].items():
                        global_row = b * 4 + lj
                        V[global_row][col] += coeff_v * coeff_b
            
            # W contribution
            # W22[li, s] is the coefficient for C_local[li]
            # C_local is the result of X*Y.
            # C_global is formed by C_combos.
            # C_block[li] = sum_{m} C_combos[m_block][k] * M_k[li]
            # So for a fixed local li, the contribution to W is:
            # For each block c in C_combos:
            #   coeff_c = C_combos[c][k]
            #   global_row = c * 4 + li
            #   W[global_row][col] += coeff_c * W22[li, s]
            
            for li in range(4):
                coeff_w_local = W22[li, s]
                if coeff_w_local != 0:
                    for c, coeff_c in C_combos.items():
                        if k in coeff_c:
                            global_row = c * 4 + li
                            W[global_row][col] += coeff_c[k] * coeff_w_local
            
            col += 1
    
    # Part 2: 4x4x5 Schoolbook
    # A is 4x4, B2 is 4x5.
    # U rows 0..15
    # V rows 16..35 (corresponding to B2)
    # W rows 16..35 (corresponding to C2)
    
    for i in range(n):
        for j in range(m):
            for k in range(p2):
                # A index: i*m + j
                # B2 index: j*p2 + k -> Global B index: (p1 + k)*m + j? No.
                # B is m x p. B[j, p1+k]. Global V row: j*p + (p1+k)
                # C2 index: i*p2 + k -> Global C index: i*p + (p1+k). W row: (p1+k)*n + i
                
                u_row = i * m + j
                v_row = j * p + (p1 + k)
                w_row = (p1 + k) * n + i
                
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
