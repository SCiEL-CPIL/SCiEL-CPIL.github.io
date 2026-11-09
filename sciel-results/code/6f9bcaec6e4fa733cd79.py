import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (16, 56)
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
    
    t0 = time.time()
    DEADLINE = 145.0
    
    col_offset = 0
    for t in range(4):
        for s in range(2):
            for p_idx in range(7):
                r = col_offset
                x_coeffs, y_coeffs, z_coeffs = strassen_defs[p_idx]
                
                for x_idx, val in x_coeffs.items():
                    u_idx = [2*s, 2*s+1, 4+2*s, 5+2*s][x_idx]
                    U[u_idx][r] += val
                
                for y_idx, val in y_coeffs.items():
                    v_idx = [16*s + 2*t, 16*s + 2*t+1, 16*s + 8 + 2*t, 16*s + 8 + 2*t+1][y_idx]
                    V[v_idx][r] += val
                
                for z_idx, val in z_coeffs.items():
                    w_idx = [4*t, 4*t+2, 4*t+1, 4*t+3][z_idx]
                    W[w_idx][r] += val
                
                col_offset += 1
                
                if time.time() - t0 > DEADLINE:
                    return {"U": U, "V": V, "W": W}
                
    return {"U": U, "V": V, "W": W}
