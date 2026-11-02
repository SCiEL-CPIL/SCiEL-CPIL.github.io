import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
         n, m, p = DIMENSIONS
         rank = 28
         U = [[0] * rank for _ in range(n * m)]
         V = [[0] * rank for _ in range(m * p)]
         W = [[0] * rank for _ in range(p * n)]
         
         # Strassen terms for 2x2
         # u_coeffs: list of (index, coeff) for U
         # v_coeffs: list of (index, coeff) for V
         # w_coeffs: list of (index, coeff) for W
         
         # We define the 7 terms for a generic block
         # Indices for U are 0..3
         # Indices for V depend on s
         # Indices for W depend on s
         
         for s in range(4):
             k0 = 2 * s
             k1 = 2 * s + 1
             
             # V indices
             v_e = 0 * 8 + k0
             v_f = 0 * 8 + k1
             v_g = 1 * 8 + k0
             v_h = 1 * 8 + k1
             
             # W indices
             w_c00 = k0 * 2 + 0
             w_c01 = k1 * 2 + 0
             w_c10 = k0 * 2 + 1
             w_c11 = k1 * 2 + 1
             
             # Term 1
             r = s * 7 + 0
             U[0][r] += 1
             U[3][r] += 1
             V[v_e][r] += 1
             V[v_h][r] += 1
             W[w_c00][r] += 1
             W[w_c11][r] += 1
             
             # Term 2
             r = s * 7 + 1
             U[2][r] += 1
             U[3][r] += 1
             V[v_e][r] += 1
             W[w_c10][r] += 1
             W[w_c11][r] += -1
             
             # Term 3
             r = s * 7 + 2
             U[0][r] += 1
             V[v_f][r] += 1
             V[v_h][r] += -1
             W[w_c01][r] += 1
             W[w_c11][r] += 1
             
             # Term 4
             r = s * 7 + 3
             U[3][r] += 1
             V[v_g][r] += 1
             V[v_e][r] += -1
             W[w_c00][r] += 1
             W[w_c10][r] += 1
             
             # Term 5
             r = s * 7 + 4
             U[0][r] += 1
             U[1][r] += 1
             V[v_h][r] += 1
             W[w_c00][r] += -1
             W[w_c01][r] += 1
             
             # Term 6
             r = s * 7 + 5
             U[2][r] += 1
             U[0][r] += -1
             V[v_e][r] += 1
             V[v_f][r] += 1
             W[w_c11][r] += 1
             
             # Term 7
             r = s * 7 + 6
             U[1][r] += 1
             U[3][r] += -1
             V[v_g][r] += 1
             V[v_h][r] += 1
             W[w_c00][r] += 1
             
         return {"U": U, "V": V, "W": W}
