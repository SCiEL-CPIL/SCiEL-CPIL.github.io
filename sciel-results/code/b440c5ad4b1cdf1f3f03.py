import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to represent a linear form as a dict {index: coeff}
    # Index for A is (i, j) -> i*m+j
    # Index for B is (i, j) -> i*m+j
    
    # Outer Strassen definitions
    # X_k entries as linear forms on A
    # Y_k entries as linear forms on B
    
    # Define blocks A_00, A_01, A_10, A_11
    # A_00 indices: (0,0), (0,1), (1,0), (1,1)
    # A_01 indices: (0,2), (0,3), (1,2), (1,3)
    # A_10 indices: (2,0), (2,1), (3,0), (3,1)
    # A_11 indices: (2,2), (2,3), (3,2), (3,3)
    
    # We will define X_k[r][c] and Y_k[r][c] as dicts
    # r, c in {0,1} for the 2x2 local block
    
    def get_A_block(br, bc, lr, lc):
        # br, bc: block row/col (0 or 1)
        # lr, lc: local row/col (0 or 1)
        return (2*br + lr) * m + (2*bc + lc)
        
    def get_B_block(br, bc, lr, lc):
        return (2*br + lr) * m + (2*bc + lc)

    # X_k and Y_k definitions
    # k=0: X = A00 + A11, Y = B00 + B11
    # k=1: X = A10 + A11, Y = B00
    # k=2: X = A00, Y = B01 - B11
    # k=3: X = A11, Y = B10 - B00
    # k=4: X = A00 + A01, Y = B11
    # k=5: X = A10 - A00, Y = B00 + B01
    # k=6: X = A01 - A11, Y = B10 + B11
    
    # We store X_k[r][c] and Y_k[r][c] as lists of (coeff, index)
    # But since they are sparse, dicts are fine.
    
    Xs = []
    Ys = []
    
    for k in range(7):
        Xk = [[{} for _ in range(2)] for _ in range(2)]
        Yk = [[{} for _ in range(2)] for _ in range(2)]
        
        if k == 0:
            # X = A00 + A11
            for r in range(2):
                for c in range(2):
                    idx1 = get_A_block(0, 0, r, c)
                    idx2 = get_A_block(1, 1, r, c)
                    Xk[r][c][idx1] = 1
                    Xk[r][c][idx2] = 1
            # Y = B00 + B11
            for r in range(2):
                for c in range(2):
                    idx1 = get_B_block(0, 0, r, c)
                    idx2 = get_B_block(1, 1, r, c)
                    Yk[r][c][idx1] = 1
                    Yk[r][c][idx2] = 1
        elif k == 1:
            # X = A10 + A11
            for r in range(2):
                for c in range(2):
                    idx1 = get_A_block(1, 0, r, c)
                    idx2 = get_A_block(1, 1, r, c)
                    Xk[r][c][idx1] = 1
                    Xk[r][c][idx2] = 1
            # Y = B00
            for r in range(2):
                for c in range(2):
                    idx1 = get_B_block(0, 0, r, c)
                    Yk[r][c][idx1] = 1
        elif k == 2:
            # X = A00
            for r in range(2):
                for c in range(2):
                    idx1 = get_A_block(0, 0, r, c)
                    Xk[r][c][idx1] = 1
            # Y = B01 - B11
            for r in range(2):
                for c in range(2):
                    idx1 = get_B_block(0, 1, r, c)
                    idx2 = get_B_block(1, 1, r, c)
                    Yk[r][c][idx1] = 1
                    Yk[r][c][idx2] = -1
        elif k == 3:
            # X = A11
            for r in range(2):
                for c in range(2):
                    idx1 = get_A_block(1, 1, r, c)
                    Xk[r][c][idx1] = 1
            # Y = B10 - B00
            for r in range(2):
                for c in range(2):
                    idx1 = get_B_block(1, 0, r, c)
                    idx2 = get_B_block(0, 0, r, c)
                    Yk[r][c][idx1] = 1
                    Yk[r][c][idx2] = -1
        elif k == 4:
            # X = A00 + A01
            for r in range(2):
                for c in range(2):
                    idx1 = get_A_block(0, 0, r, c)
                    idx2 = get_A_block(0, 1, r, c)
                    Xk[r][c][idx1] = 1
                    Xk[r][c][idx2] = 1
            # Y = B11
            for r in range(2):
                for c in range(2):
                    idx1 = get_B_block(1, 1, r, c)
                    Yk[r][c][idx1] = 1
        elif k == 5:
            # X = A10 - A00
            for r in range(2):
                for c in range(2):
                    idx1 = get_A_block(1, 0, r, c)
                    idx2 = get_A_block(0, 0, r, c)
                    Xk[r][c][idx1] = 1
                    Xk[r][c][idx2] = -1
            # Y = B00 + B01
            for r in range(2):
                for c in range(2):
                    idx1 = get_B_block(0, 0, r, c)
                    idx2 = get_B_block(0, 1, r, c)
                    Yk[r][c][idx1] = 1
                    Yk[r][c][idx2] = 1
        elif k == 6:
            # X = A01 - A11
            for r in range(2):
                for c in range(2):
                    idx1 = get_A_block(0, 1, r, c)
                    idx2 = get_A_block(1, 1, r, c)
                    Xk[r][c][idx1] = 1
                    Xk[r][c][idx2] = -1
            # Y = B10 + B11
            for r in range(2):
                for c in range(2):
                    idx1 = get_B_block(1, 0, r, c)
                    idx2 = get_B_block(1, 1, r, c)
                    Yk[r][c][idx1] = 1
                    Yk[r][c][idx2] = 1
                    
        Xs.append(Xk)
        Ys.append(Yk)
        
    # Inner Strassen definitions
    # l=0: P = X00 + X11, Q = Y00 + Y11
    # l=1: P = X10 + X11, Q = Y00
    # l=2: P = X00, Q = Y01 - Y11
    # l=3: P = X11, Q = Y10 - Y00
    # l=4: P = X00 + X01, Q = Y11
    # l=5: P = X10 - X00, Q = Y00 + Y01
    # l=6: P = X01 - X11, Q = Y10 + Y11
    
    # Output reconstruction coefficients alpha[I][J][k]
    alpha = [
        [1, 0, 0, 1, -1, 0, 1], # I=0, J=0
        [0, 0, 1, 0, 1, 0, 0],  # I=0, J=1
        [0, 1, 0, 1, 0, 0, 0],  # I=1, J=0
        [1, -1, 1, 0, 0, 1, 0]  # I=1, J=1
    ]
    
    # M_k[r][c] coefficients beta[r][c][l]
    # M00 = S0 + S3 - S4 + S6
    # M01 = S2 + S4
    # M10 = S1 + S3
    # M11 = S0 - S1 + S2 + S5
    beta = [
        [ # r=0
            [1, 0, 0, 1, -1, 0, 1], # c=0
            [0, 0, 1, 0, 1, 0, 0]   # c=1
        ],
        [ # r=1
            [0, 1, 0, 1, 0, 0, 0],  # c=0
            [1, -1, 1, 0, 0, 1, 0]  # c=1
        ]
    ]
    
    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            
            # Compute u (linear form on A) and v (linear form on B)
            u_dict = {}
            v_dict = {}
            
            # Determine P and Q definitions for this l
            # P is linear in X_k entries, Q is linear in Y_k entries
            
            # Helper to add linear forms
            def add_form(target, source, coeff):
                for idx, val in source.items():
                    target[idx] = target.get(idx, 0) + coeff * val
            
            if l == 0:
                # P = X00 + X11
                add_form(u_dict, Xs[k][0][0], 1)
                add_form(u_dict, Xs[k][1][1], 1)
                # Q = Y00 + Y11
                add_form(v_dict, Ys[k][0][0], 1)
                add_form(v_dict, Ys[k][1][1], 1)
            elif l == 1:
                # P = X10 + X11
                add_form(u_dict, Xs[k][1][0], 1)
                add_form(u_dict, Xs[k][1][1], 1)
                # Q = Y00
                add_form(v_dict, Ys[k][0][0], 1)
            elif l == 2:
                # P = X00
                add_form(u_dict, Xs[k][0][0], 1)
                # Q = Y01 - Y11
                add_form(v_dict, Ys[k][0][1], 1)
                add_form(v_dict, Ys[k][1][1], -1)
            elif l == 3:
                # P = X11
                add_form(u_dict, Xs[k][1][1], 1)
                # Q = Y10 - Y00
                add_form(v_dict, Ys[k][1][0], 1)
                add_form(v_dict, Ys[k][0][0], -1)
            elif l == 4:
                # P = X00 + X01
                add_form(u_dict, Xs[k][0][0], 1)
                add_form(u_dict, Xs[k][0][1], 1)
                # Q = Y11
                add_form(v_dict, Ys[k][1][1], 1)
            elif l == 5:
                # P = X10 - X00
                add_form(u_dict, Xs[k][1][0], 1)
                add_form(u_dict, Xs[k][0][0], -1)
                # Q = Y00 + Y01
                add_form(v_dict, Ys[k][0][0], 1)
                add_form(v_dict, Ys[k][0][1], 1)
            elif l == 6:
                # P = X01 - X11
                add_form(u_dict, Xs[k][0][1], 1)
                add_form(u_dict, Xs[k][1][1], -1)
                # Q = Y10 + Y11
                add_form(v_dict, Ys[k][1][0], 1)
                add_form(v_dict, Ys[k][1][1], 1)
                
            # Fill U and V
            for idx, val in u_dict.items():
                if val != 0:
                    U[idx][col] = val
            for idx, val in v_dict.items():
                if val != 0:
                    V[idx][col] = val
                    
            # Fill W
            # For each output index i, j
            for i in range(n):
                for j in range(p):
                    I = i // 2
                    J = j // 2
                    r = i % 2
                    c = j % 2
                    
                    coeff = alpha[I][J][k] * beta[r][c][l]
                    if coeff != 0:
                        W[i * n + j][col] = coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
