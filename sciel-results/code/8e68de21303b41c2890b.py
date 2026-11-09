import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # u[s][I][J] = coefficient of A[I][J] in s-th Strassen product
    # v[s][J][K] = coefficient of B[J][K] in s-th Strassen product
    # w[s][I][K] = coefficient of P_s in C[I][K]
    
    u_2x2 = [
        [[1, 0], [0, 1]],   # P1: (A00+A11)
        [[0, 0], [1, 1]],   # P2: (A10+A11)
        [[1, 0], [0, 0]],   # P3: A00
        [[0, 0], [0, 1]],   # P4: A11
        [[1, 1], [0, 0]],   # P5: (A00+A01)
        [[-1, 0], [1, 0]],  # P6: (A10-A00)
        [[0, 1], [0, -1]],  # P7: (A01-A11)
    ]
    v_2x2 = [
        [[1, 0], [0, 1]],   # P1: (B00+B11)
        [[1, 0], [0, 0]],   # P2: B00
        [[0, 1], [0, -1]],  # P3: (B01-B11)
        [[0, -1], [1, 0]],  # P4: (B10-B00)
        [[0, 0], [0, 1]],   # P5: B11
        [[1, 1], [0, 0]],   # P6: (B00+B01)
        [[0, 0], [1, 1]],   # P7: (B10+B11)
    ]
    w_2x2 = [
        [[1, 0], [0, 0]],   # C00 += P1
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ]
    # Full w_2x2 for reconstruction:
    # C00 = P1 - P2 + P3 + P6
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    w_recon = [
        [[1, 0], [0, 0]],   # P1 -> C00
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ]
    # Actually let me define w_recon properly:
    # w_recon[t][I][K] = coefficient of P_t in C[I][K]
    w_recon = [
        [[1, 0], [0, 1]],   # P1: C00+=1, C11+=1
        [[-1, 0], [0, -1]], # P2: C00+=-1, C11+=-1
        [[1, 1], [0, 1]],   # P3: C00+=1, C01+=1, C11+=1
        [[0, 0], [1, 0]],   # P4: C10+=1
        [[0, 1], [0, 0]],   # P5: C01+=1
        [[1, 0], [0, 0]],   # P6: C00+=1
        [[0, 0], [0, 0]],   # P7: C01+=1 (wait, C01 = P3+P5, C11 = P1-P2+P3+P6)
    ]
    # Let me redo w_recon carefully:
    # C00 = P1 - P2 + P3 + P6
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    w_recon = [
        [[1, 0], [0, 1]],   # P1
        [[-1, 0], [0, -1]], # P2
        [[1, 1], [0, 1]],   # P3
        [[0, 0], [1, 0]],   # P4
        [[0, 1], [0, 0]],   # P5
        [[1, 0], [0, 0]],   # P6
        [[0, 0], [0, 0]],   # P7
    ]
    
    # Outer Strassen: 4x4 matrix A, B partitioned into 2x2 blocks of 2x2 matrices
    # A[I][J] is a 2x2 block, B[J][K] is a 2x2 block
    # u_outer[s][I][J] = coeff of A[I][J] in L_s
    # v_outer[s][J][K] = coeff of B[J][K] in R_s
    
    u_outer = [
        [[1, 0], [0, 1]],   # L0: A00+A11
        [[0, 0], [1, 1]],   # L1: A10+A11
        [[1, 0], [0, 0]],   # L2: A00
        [[0, 0], [0, 1]],   # L3: A11
        [[1, 1], [0, 0]],   # L4: A00+A01
        [[-1, 0], [1, 0]],  # L5: A10-A00
        [[0, 1], [0, -1]],  # L6: A01-A11
    ]
    v_outer = [
        [[1, 0], [0, 1]],   # R0: B00+B11
        [[1, 0], [0, 0]],   # R1: B00
        [[0, 1], [0, -1]],  # R2: B01-B11
        [[0, -1], [1, 0]],  # R3: B10-B00
        [[0, 0], [0, 1]],   # R4: B11
        [[1, 1], [0, 0]],   # R5: B00+B01
        [[0, 0], [1, 1]],   # R6: B10+B11
    ]
    w_outer = [
        [[1, 0], [0, 1]],   # P0: C00+=1, C11+=1
        [[-1, 0], [0, -1]], # P1: C00+=-1, C11+=-1
        [[1, 1], [0, 1]],   # P2: C00+=1, C01+=1, C11+=1
        [[0, 0], [1, 0]],   # P3: C10+=1
        [[0, 1], [0, 0]],   # P4: C01+=1
        [[1, 0], [0, 0]],   # P5: C00+=1
        [[0, 0], [0, 0]],   # P6
    ]
    
    # For each outer step s (0..6) and inner step t (0..6):
    # The product is (sum_{I,J} u_outer[s][I][J]*u_2x2[t][I][J]*A[I][J]) *
    #                (sum_{J,K} v_outer[s][J][K]*v_2x2[t][J][K]*B[J][K])
    # Contribution to C[I][K] = w_outer[s][I][K]*w_recon[t][I][K]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # U coefficients: for each A[I][J] block (2x2), coefficient is u_outer[s][I][J]*u_2x2[t][I][J]
            # A[I][J] corresponds to rows 2I,2I+1 and cols 2J,2J+1
            # U index: i*4+j where i=2I+a, j=2J+b
            for I in range(2):
                for J in range(2):
                    coeff = u_outer[s][I][J] * u_2x2[t][I][J]
                    if coeff != 0:
                        for a in range(2):
                            for b in range(2):
                                u_idx = (2 * I + a) * 4 + (2 * J + b)
                                U[u_idx][r] += coeff
            
            # V coefficients: for each B[J][K] block (2x2), coefficient is v_outer[s][J][K]*v_2x2[t][J][K]
            # B[J][K] corresponds to rows 2J,2J+1 and cols 2K,2K+1
            # V index: j*4+k where j=2J+b, k=2K+c
            for J in range(2):
                for K in range(2):
                    coeff = v_outer[s][J][K] * v_2x2[t][J][K]
                    if coeff != 0:
                        for b in range(2):
                            for c in range(2):
                                v_idx = (2 * J + b) * 4 + (2 * K + c)
                                V[v_idx][r] += coeff
            
            # W coefficients: for each C[I][K] block (2x2), coefficient is w_outer[s][I][K]*w_recon[t][I][K]
            # C[I][K] corresponds to rows 2K,2K+1 and cols 2I,2I+1 (W index: k*n+i)
            # W index: k*4+i where k=2K+c, i=2I+a
            for I in range(2):
                for K in range(2):
                    coeff = w_outer[s][I][K] * w_recon[t][I][K]
                    if coeff != 0:
                        for a in range(2):
                            for c in range(2):
                                w_idx = (2 * K + c) * 4 + (2 * I + a)
                                W[w_idx][r] += coeff
    
    return {"U": U, "V": V, "W": W}
