import time
import sys

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen coefficients for X_s (from A) and Y_s (from B)
    # s=1..7
    # X_s is sum of A_blocks with coeffs
    # Y_s is sum of B_blocks with coeffs
    # Block indices I, J in {1, 2}
    
    # c_s[I-1][J-1] for X_s
    # c_s[I-1][J-1] for Y_s (using J, K for B)
    
    # Let's define the linear combinations
    # s=1: X = A11 + A22, Y = B11 + B22
    # s=2: X = A21 + A22, Y = B11
    # s=3: X = A11,       Y = B12 - B22
    # s=4: X = A22,       Y = B21 - B11
    # s=5: X = A11 + A12, Y = B22
    # s=6: X = A21 - A11, Y = B11 + B12
    # s=7: X = A12 - A22, Y = B21 + B22
    
    # Coefficients c_x[s][I][J]
    c_x = {}
    c_y = {}
    
    def set_coeff(dict, s, I, J, val):
        if s not in dict:
            dict[s] = {}
        if I not in dict[s]:
            dict[s][I] = {}
        dict[s][I][J] = val

    # s=1
    set_coeff(c_x, 1, 1, 1, 1); set_coeff(c_x, 1, 2, 2, 1)
    set_coeff(c_y, 1, 1, 1, 1); set_coeff(c_y, 1, 2, 2, 1)
    # s=2
    set_coeff(c_x, 2, 2, 1, 1); set_coeff(c_x, 2, 2, 2, 1)
    set_coeff(c_y, 2, 1, 1, 1)
    # s=3
    set_coeff(c_x, 3, 1, 1, 1)
    set_coeff(c_y, 3, 1, 2, 1); set_coeff(c_y, 3, 2, 2, -1)
    # s=4
    set_coeff(c_x, 4, 2, 2, 1)
    set_coeff(c_y, 4, 2, 1, 1); set_coeff(c_y, 4, 1, 1, -1)
    # s=5
    set_coeff(c_x, 5, 1, 1, 1); set_coeff(c_x, 5, 1, 2, 1)
    set_coeff(c_y, 5, 2, 2, 1)
    # s=6
    set_coeff(c_x, 6, 2, 1, 1); set_coeff(c_x, 6, 1, 1, -1)
    set_coeff(c_y, 6, 1, 1, 1); set_coeff(c_y, 6, 1, 2, 1)
    # s=7
    set_coeff(c_x, 7, 1, 2, 1); set_coeff(c_x, 7, 2, 2, -1)
    set_coeff(c_y, 7, 2, 1, 1); set_coeff(c_y, 7, 2, 2, 1)

    # Inner Strassen coefficients d_t[i][j] for x and y
    # t=1..7
    # t=1: x = x11 + x22, y = y11 + y22
    # t=2: x = x21 + x22, y = y11
    # t=3: x = x11,       y = y12 - y22
    # t=4: x = x22,       y = y21 - y11
    # t=5: x = x11 + x12, y = y22
    # t=6: x = x21 - x11, y = y11 + y12
    # t=7: x = x12 - x22, y = y21 + y22
    
    d_x = {}
    d_y = {}
    
    def set_d(dict, t, i, j, val):
        if t not in dict:
            dict[t] = {}
        if i not in dict[t]:
            dict[t][i] = {}
        dict[t][i][j] = val

    # t=1
    set_d(d_x, 1, 0, 0, 1); set_d(d_x, 1, 1, 1, 1)
    set_d(d_y, 1, 0, 0, 1); set_d(d_y, 1, 1, 1, 1)
    # t=2
    set_d(d_x, 2, 1, 0, 1); set_d(d_x, 2, 1, 1, 1)
    set_d(d_y, 2, 0, 0, 1)
    # t=3
    set_d(d_x, 3, 0, 0, 1)
    set_d(d_y, 3, 0, 1, 1); set_d(d_y, 3, 1, 1, -1)
    # t=4
    set_d(d_x, 4, 1, 1, 1)
    set_d(d_y, 4, 1, 0, 1); set_d(d_y, 4, 0, 0, -1)
    # t=5
    set_d(d_x, 5, 0, 0, 1); set_d(d_x, 5, 0, 1, 1)
    set_d(d_y, 5, 1, 1, 1)
    # t=6
    set_d(d_x, 6, 1, 0, 1); set_d(d_x, 6, 0, 0, -1)
    set_d(d_y, 6, 0, 0, 1); set_d(d_y, 6, 0, 1, 1)
    # t=7
    set_d(d_x, 7, 0, 1, 1); set_d(d_x, 7, 1, 1, -1)
    set_d(d_y, 7, 1, 0, 1); set_d(d_y, 7, 1, 1, 1)

    # W coefficients
    # coeff of M_s in C_IJ
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    w_coeff = {}
    # w_coeff[s][I][J]
    set_coeff(w_coeff, 1, 1, 1, 1); set_coeff(w_coeff, 1, 2, 2, 1)
    set_coeff(w_coeff, 2, 2, 1, 1); set_coeff(w_coeff, 2, 2, 2, -1)
    set_coeff(w_coeff, 3, 1, 2, 1); set_coeff(w_coeff, 3, 2, 2, 1)
    set_coeff(w_coeff, 4, 1, 1, 1); set_coeff(w_coeff, 4, 2, 1, 1)
    set_coeff(w_coeff, 5, 1, 1, -1); set_coeff(w_coeff, 5, 1, 2, 1)
    set_coeff(w_coeff, 6, 2, 2, 1)
    set_coeff(w_coeff, 7, 1, 1, 1)

    for s in range(1, 8):
        for t in range(1, 8):
            r = (s - 1) * 7 + (t - 1)
            
            # Compute U
            for i in range(2):
                for j in range(2):
                    val_d = d_x[t].get(i, {}).get(j, 0)
                    if val_d == 0: continue
                    for I in range(1, 3):
                        for J in range(1, 3):
                            val_c = c_x[s].get(I, {}).get(J, 0)
                            if val_c == 0: continue
                            gi = (I - 1) * 2 + i
                            gj = (J - 1) * 2 + j
                            U[gi * 4 + gj][r] += val_d * val_c
                            
            # Compute V
            for i in range(2):
                for j in range(2):
                    val_d = d_y[t].get(i, {}).get(j, 0)
                    if val_d == 0: continue
                    for J in range(1, 3):
                        for K in range(1, 3):
                            val_c = c_y[s].get(J, {}).get(K, 0)
                            if val_c == 0: continue
                            gj = (J - 1) * 2 + i
                            gk = (K - 1) * 2 + j
                            V[gj * 4 + gk][r] += val_d * val_c
                            
            # Compute W
            for I in range(1, 3):
                for J in range(1, 3):
                    val_w = w_coeff[s].get(I, {}).get(J, 0)
                    if val_w == 0: continue
                    for gi in range(2):
                        for gj in range(2):
                            # gi is row of C (i), gj is col of C (k)
                            # Wait, W index is k*n + i.
                            # i is row of C, k is col of C.
                            # C_IJ corresponds to i in block I, k in block J.
                            # i_global = (I-1)*2 + gi
                            # k_global = (J-1)*2 + gj
                            gi_g = (I - 1) * 2 + gi
                            gj_g = (J - 1) * 2 + gj
                            W[gj_g * 4 + gi_g][r] = val_w

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
