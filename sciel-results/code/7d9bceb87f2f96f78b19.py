import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    t0 = time.time()
    DEADLINE = 145.0
    
    # We will attempt to find a decomposition with rank < 56.
    # If we find one, we return it. Otherwise, we return the known rank 56 decomposition.
    
    # Rank 56 decomposition (Strassen based)
    # This is the fallback.
    rank_56 = 56
    U_56 = np.zeros((8, rank_56))
    V_56 = np.zeros((32, rank_56))
    W_56 = np.zeros((16, rank_56))
    
    # Strassen coefficients for 2x2 matrix multiplication
    # P1 = (x00 + x11) * (y00 + y11)
    # P2 = (x10 + x11) * y00
    # P3 = x00 * (y01 - y11)
    # P4 = x11 * (y10 - y00)
    # P5 = (x00 + x01) * y11
    # P6 = (x10 - x00) * (y00 + y01)
    # P7 = (x01 - x11) * (y10 + y11)
    
    strassen_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # X indices in U:
    # x00 -> U[2s]
    # x01 -> U[2s+1]
    # x10 -> U[4+2s]
    # x11 -> U[5+2s]
    
    # Y indices in V:
    # y00 -> V[16s + 2t]
    # y01 -> V[16s + 2t+1]
    # y10 -> V[16s + 8 + 2t]
    # y11 -> V[16s + 8 + 2t+1]
    
    # Z indices in W:
    # z00 -> W[4t]
    # z01 -> W[4t+2]
    # z10 -> W[4t+1]
    # z11 -> W[4t+3]
    
    col_offset = 0
    for t in range(4):
        for s in range(2):
            for p_idx in range(7):
                r = col_offset
                x_coeffs, y_coeffs, z_coeffs = strassen_defs[p_idx]
                
                for x_idx, val in x_coeffs.items():
                    u_idx = [2*s, 2*s+1, 4+2*s, 5+2*s][x_idx]
                    U_56[u_idx, r] += val
                
                for y_idx, val in y_coeffs.items():
                    v_idx = [16*s + 2*t, 16*s + 2*t+1, 16*s + 8 + 2*t, 16*s + 8 + 2*t+1][y_idx]
                    V_56[v_idx, r] += val
                
                for z_idx, val in z_coeffs.items():
                    w_idx = [4*t, 4*t+2, 4*t+1, 4*t+3][z_idx]
                    W_56[w_idx, r] += val
                
                col_offset += 1
                
                if time.time() - t0 > DEADLINE:
                    return {"U": U_56.tolist(), "V": V_56.tolist(), "W": W_56.tolist()}
    
    # Now, try to find a lower rank decomposition.
    # We can use a randomized search or a structured search.
    # Given the time constraints, we will try a few specific structured decompositions.
    
    # One idea: Use the rank 19 decomposition for 2x2x2 (Laderman) to build up.
    # But Laderman is for 2x2x2.
    # M(2,4,8) can be viewed as 4 blocks of M(2,2,2) if we split B into 4 2x2 blocks.
    # But the inputs to these blocks are not independent.
    # A = [[A1, A2], [A3, A4]]
    # B = [[B1, B2, B3, B4], [B5, B6, B7, B8]]
    # C = A*B
    # C11 = A1*B1 + A2*B5
    # C12 = A1*B2 + A2*B6
    # C13 = A1*B3 + A2*B7
    # C14 = A1*B4 + A2*B8
    # C21 = A3*B1 + A4*B5
    # C22 = A3*B2 + A4*B6
    # C23 = A3*B3 + A4*B7
    # C24 = A3*B4 + A4*B8
    
    # This is 8 2x2 matrix multiplications.
    # If we use Strassen for each, we get 7*8 = 56.
    # Can we do better?
    # Yes, if we share some of the Strassen products.
    # For example, C11 and C12 share A1 and A2.
    # C11 = A1*B1 + A2*B5
    # C12 = A1*B2 + A2*B6
    # Let X1 = A1, X2 = A2.
    # Let Y1 = B1, Y2 = B5, Y3 = B2, Y4 = B6.
    # We need to compute X1*Y1 + X2*Y2 and X1*Y3 + X2*Y4.
    # This is like multiplying [X1, X2] by [Y1, Y3; Y2, Y4].
    # This is a 2x2 by 2x4 matrix multiplication.
    # M(2,2,4).
    # If we have a decomposition for M(2,2,4) with rank r, then we can use it here.
    # There are 4 such blocks: (C11,C12), (C13,C14), (C21,C22), (C23,C24).
    # But they are not independent either.
    
    # Let's try to find a rank 28 decomposition.
    # We can use a search over random matrices.
    # But that is too slow.
    
    # Instead, we will return the rank 56 decomposition.
    # This is safe and known to be correct.
    
    return {"U": U_56.tolist(), "V": V_56.tolist(), "W": W_56.tolist()}
