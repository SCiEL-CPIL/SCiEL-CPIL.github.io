import numpy as np
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the multiplication into two independent blocks:
    # C = A * B
    # B is 3x4, split into B_L (3x2) and B_R (3x2)
    # C_L = A * B_L (2x2)
    # C_R = A * B_R (2x2)
    # Each A * B_i is a 2x3 by 3x2 multiplication.
    # We use a rank 11 decomposition for 2x3 by 3x2.
    # Total rank = 11 + 11 = 22.
    
    rank = 22
    
    # Initialize U, V, W
    # U: (2*3, 22) = (6, 22)
    # V: (3*4, 22) = (12, 22)
    # W: (4*2, 22) = (8, 22)
    
    U = np.zeros((6, rank), dtype=int)
    V = np.zeros((12, rank), dtype=int)
    W = np.zeros((8, rank), dtype=int)
    
    # Helper to add a rank-1 component
    def add_component(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w, offset):
        U[u_idx, offset] += coeff_u
        V[v_idx, offset] += coeff_v
        W[w_idx, offset] += coeff_w
        
    # Define the rank 11 decomposition for 2x3 by 3x2
    # A is 2x3, B is 3x2.
    # A = [[a1, a2, a3], [d1, d2, d3]]
    # B = [[b11, b12], [b21, b22], [b31, b32]]
    # C = A * B = [[c11, c12], [c21, c22]]
    
    # Split A into A1 (2x1: [a1, d1]) and A2 (2x2: [[a2, a3], [d2, d3]])
    # Split B into B1 (1x2: [b11, b12]) and B2 (2x2: [[b21, b22], [b31, b32]])
    # C = A1 * B1 + A2 * B2
    
    # Part 1: A1 * B1 (Rank 4)
    # c11 += a1 * b11
    # c12 += a1 * b12
    # c21 += d1 * b11
    # c22 += d1 * b12
    
    # Part 2: A2 * B2 (Rank 7, Strassen)
    # Let A2 = [[a2, a3], [d2, d3]]
    # Let B2 = [[b21, b22], [b31, b32]]
    # Standard Strassen for 2x2:
    # m1 = (a2 + d2) * (b21 + b31)
    # m2 = (d2 + d3) * b31
    # m3 = a2 * (b22 - b32)
    # m4 = d3 * (b21 + b22)
    # m5 = (a2 + a3) * b32
    # m6 = (a3 - d2) * (b21 + b22)
    # m7 = (d2 - d3) * (b31 + b32)
    # c11 += m1 + m4 - m5 - m7
    # c12 += m3 + m5
    # c21 += m2 + m4
    # c22 += m1 - m2 + m3 + m6
    
    # Indices in U, V, W for the first block (offset 0)
    # U indices for A:
    # a1: 0, a2: 1, a3: 2
    # d1: 3, d2: 4, d3: 5
    
    # V indices for B_L (first 2 cols of B):
    # b11: 0, b12: 1
    # b21: 2, b22: 3
    # b31: 4, b32: 5
    
    # W indices for C_L (first 2 cols of C):
    # c11: 0, c12: 1
    # c21: 2, c22: 3
    
    # --- Block L (offset 0) ---
    offset = 0
    
    # A1 * B1
    # 1. a1 * b11 -> c11
    U[0, offset] = 1
    V[0, offset] = 1
    W[0, offset] = 1
    offset += 1
    
    # 2. a1 * b12 -> c12
    U[0, offset] = 1
    V[1, offset] = 1
    W[1, offset] = 1
    offset += 1
    
    # 3. d1 * b11 -> c21
    U[3, offset] = 1
    V[0, offset] = 1
    W[2, offset] = 1
    offset += 1
    
    # 4. d1 * b12 -> c22
    U[3, offset] = 1
    V[1, offset] = 1
    W[3, offset] = 1
    offset += 1
    
    # A2 * B2 (Strassen)
    # m1 = (a2 + d2) * (b21 + b31)
    U[1, offset] = 1
    U[4, offset] = 1
    V[2, offset] = 1
    V[4, offset] = 1
    # c11 += m1, c22 += m1
    W[0, offset] = 1
    W[3, offset] = 1
    offset += 1
    
    # m2 = (d2 + d3) * b31
    U[4, offset] = 1
    U[5, offset] = 1
    V[4, offset] = 1
    # c21 += m2, c22 -= m2
    W[2, offset] = 1
    W[3, offset] = -1
    offset += 1
    
    # m3 = a2 * (b22 - b32)
    U[1, offset] = 1
    V[3, offset] = 1
    V[5, offset] = -1
    # c12 += m3, c22 += m3
    W[1, offset] = 1
    W[3, offset] = 1
    offset += 1
    
    # m4 = d3 * (b21 + b22)
    U[5, offset] = 1
    V[2, offset] = 1
    V[3, offset] = 1
    # c11 += m4, c21 += m4
    W[0, offset] = 1
    W[2, offset] = 1
    offset += 1
    
    # m5 = (a2 + a3) * b32
    U[1, offset] = 1
    U[2, offset] = 1
    V[5, offset] = 1
    # c11 -= m5, c12 += m5
    W[0, offset] = -1
    W[1, offset] = 1
    offset += 1
    
    # m6 = (a3 - d2) * (b21 + b22)
    U[2, offset] = 1
    U[4, offset] = -1
    V[2, offset] = 1
    V[3, offset] = 1
    # c22 += m6
    W[3, offset] = 1
    offset += 1
    
    # m7 = (d2 - d3) * (b31 + b32)
    U[4, offset] = 1
    U[5, offset] = -1
    V[4, offset] = 1
    V[5, offset] = 1
    # c11 -= m7
    W[0, offset] = -1
    offset += 1
    
    # --- Block R (offset 11) ---
    # Same structure, but V and W indices shifted
    # V indices for B_R (last 2 cols of B):
    # b13: 6, b14: 7
    # b23: 8, b24: 9
    # b33: 10, b34: 11
    
    # W indices for C_R (last 2 cols of C):
    # c13: 4, c14: 5
    # c23: 6, c24: 7
    
    offset = 11
    
    # A1 * B1 (using B_R columns)
    # 1. a1 * b13 -> c13
    U[0, offset] = 1
    V[6, offset] = 1
    W[4, offset] = 1
    offset += 1
    
    # 2. a1 * b14 -> c14
    U[0, offset] = 1
    V[7, offset] = 1
    W[5, offset] = 1
    offset += 1
    
    # 3. d1 * b13 -> c23
    U[3, offset] = 1
    V[6, offset] = 1
    W[6, offset] = 1
    offset += 1
    
    # 4. d1 * b14 -> c24
    U[3, offset] = 1
    V[7, offset] = 1
    W[7, offset] = 1
    offset += 1
    
    # A2 * B2 (Strassen)
    # m1 = (a2 + d2) * (b23 + b33)
    U[1, offset] = 1
    U[4, offset] = 1
    V[8, offset] = 1
    V[10, offset] = 1
    W[4, offset] = 1
    W[7, offset] = 1
    offset += 1
    
    # m2 = (d2 + d3) * b33
    U[4, offset] = 1
    U[5, offset] = 1
    V[10, offset] = 1
    W[6, offset] = 1
    W[7, offset] = -1
    offset += 1
    
    # m3 = a2 * (b24 - b34)
    U[1, offset] = 1
    V[9, offset] = 1
    V[11, offset] = -1
    W[5, offset] = 1
    W[7, offset] = 1
    offset += 1
    
    # m4 = d3 * (b23 + b24)
    U[5, offset] = 1
    V[8, offset] = 1
    V[9, offset] = 1
    W[4, offset] = 1
    W[6, offset] = 1
    offset += 1
    
    # m5 = (a2 + a3) * b34
    U[1, offset] = 1
    U[2, offset] = 1
    V[11, offset] = 1
    W[4, offset] = -1
    W[5, offset] = 1
    offset += 1
    
    # m6 = (a3 - d2) * (b23 + b24)
    U[2, offset] = 1
    U[4, offset] = -1
    V[8, offset] = 1
    V[9, offset] = 1
    W[7, offset] = 1
    offset += 1
    
    # m7 = (d2 - d3) * (b33 + b34)
    U[4, offset] = 1
    U[5, offset] = -1
    V[10, offset] = 1
    V[11, offset] = 1
    W[4, offset] = -1
    offset += 1
    
    return {"U": U, "V": V, "W": W}
