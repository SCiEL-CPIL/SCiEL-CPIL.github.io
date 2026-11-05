"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 54 construction using Strassen on 2x2 blocks
    # A: 2x5, B: 5x6
    # Partition A cols: [0,1], [2,3], [4]
    # Partition B rows: [0,1], [2,3], [4]
    # Partition B cols: [0,1], [2,3], [4,5]
    
    rank = 54
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col_idx = 0
    
    # Strassen coefficients for 2x2 multiplication
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y01-y11)
    # m4 = x11(y10-y00)
    # m5 = (x00+x01)y11
    # m6 = (x10-x00)(y00+y01)
    # m7 = (x01-x11)(y10+y11)
    
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6
    
    # Helper to add a Strassen block
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col_idx
        # a_rows, a_cols, b_rows, b_cols are lists of 2 indices
        # c_rows, c_cols are lists of 2 indices
        
        # Extract local indices
        # A local: 0->(a_rows[0], a_cols[0]), 1->(a_rows[0], a_cols[1]), 2->(a_rows[1], a_cols[0]), 3->(a_rows[1], a_cols[1])
        # Let's map local (r,c) to global index
        # local 0: i=a_rows[0], j=a_cols[0]
        # local 1: i=a_rows[0], j=a_cols[1]
        # local 2: i=a_rows[1], j=a_cols[0]
        # local 3: i=a_rows[1], j=a_cols[1]
        
        # B local: 0->(b_rows[0], b_cols[0]), 1->(b_rows[0], b_cols[1]), 2->(b_rows[1], b_cols[0]), 3->(b_rows[1], b_cols[1])
        
        # C local: 0->(c_rows[0], c_cols[0]), 1->(c_rows[0], c_cols[1]), 2->(c_rows[1], c_cols[0]), 3->(c_rows[1], c_cols[1])
        # Note: C global index is k*n + i.
        # c_rows are i indices? No, c_rows in my previous thought process were i indices.
        # Let's stick to: c_rows = i indices, c_cols = k indices.
        
        a_idx = [
            a_rows[0]*m + a_cols[0],
            a_rows[0]*m + a_cols[1],
            a_rows[1]*m + a_cols[0],
            a_rows[1]*m + a_cols[1]
        ]
        b_idx = [
            b_rows[0]*p + b_cols[0],
            b_rows[0]*p + b_cols[1],
            b_rows[1]*p + b_cols[0],
            b_rows[1]*p + b_cols[1]
        ]
        c_idx = [
            c_cols[0]*n + c_rows[0],
            c_cols[1]*n + c_rows[0],
            c_cols[0]*n + c_rows[1],
            c_cols[1]*n + c_rows[1]
        ]
        
        # Define the 7 products
        # Each product is a list of (a_local, coeff) and (b_local, coeff)
        # And for each c_local, the sum of coeffs
        
        # m1: A: 0(+1), 3(+1). B: 0(+1), 3(+1)
        # m2: A: 2(+1), 3(+1). B: 0(+1)
        # m3: A: 0(+1). B: 1(+1), 3(-1)
        # m4: A: 3(+1). B: 2(+1), 0(-1)
        # m5: A: 0(+1), 1(+1). B: 3(+1)
        # m6: A: 2(+1), 0(-1). B: 0(+1), 1(+1)
        # m7: A: 1(+1), 3(-1). B: 2(+1), 3(+1)
        
        strassen_prods = [
            # (a_coeffs, b_coeffs)
            # a_coeffs is dict {local_idx: coeff}
            ({0: 1, 3: 1}, {0: 1, 3: 1}), # m1
            ({2: 1, 3: 1}, {0: 1}),       # m2
            ({0: 1}, {1: 1, 3: -1}),      # m3
            ({3: 1}, {2: 1, 0: -1}),      # m4
            ({0: 1, 1: 1}, {3: 1}),       # m5
            ({2: 1, 0: -1}, {0: 1, 1: 1}),# m6
            ({1: 1, 3: -1}, {2: 1, 3: 1}) # m7
        ]
        
        # Output combinations
        # z00 = m1 + m4 - m5 + m7
        # z01 = m3 + m5
        # z10 = m2 + m4
        # z11 = m1 - m2 + m3 + m6
        
        out_combos = [
            {0: 1, 3: 1, 4: -1, 6: 1}, # z00 (m1, m4, m5, m7)
            {2: 1, 4: 1},               # z01 (m3, m5)
            {1: 1, 3: 1},               # z10 (m2, m4)
            {0: 1, 1: -1, 2: 1, 5: 1}   # z11 (m1, m2, m3, m6)
        ]
        
        for t in range(7):
            a_coeffs, b_coeffs = strassen_prods[t]
            
            for a_loc, coeff in a_coeffs.items():
                U[a_idx[a_loc]][col_idx + t] += coeff
            
            for b_loc, coeff in b_coeffs.items():
                V[b_idx[b_loc]][col_idx + t] += coeff
            
            # Update W
            for c_loc in range(4):
                coeff = out_combos[c_loc].get(t, 0)
                if coeff != 0:
                    W[c_idx[c_loc]][col_idx + t] += coeff
        
        col_idx += 7
    
    # Add 6 Strassen blocks
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen([0, 1], [0, 1], [0, 1], [4, 5], [0, 1], [4, 5])
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen([0, 1], [2, 3], [2, 3], [4, 5], [0, 1], [4, 5])
    
    # Block 7: A[0:2, 4:5] * B[4:5, 0:6] -> C[0:2, 0:6]
    # Schoolbook: 2x1x6 = 12 terms
    # A cols: [4]
    # B rows: [4]
    # B cols: [0, 1, 2, 3, 4, 5]
    for i in range(2):
        for k in range(6):
            a_idx = i * m + 4
            b_idx = 4 * p + k
            c_idx = k * n + i
            U[a_idx][col_idx] = 1
            V[b_idx][col_idx] = 1
            W[c_idx][col_idx] = 1
            col_idx += 1
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(nm)]
        V_fb = [[0] * rank_fb for _ in range(mp)]
        W_fb = [[0] * rank_fb for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
