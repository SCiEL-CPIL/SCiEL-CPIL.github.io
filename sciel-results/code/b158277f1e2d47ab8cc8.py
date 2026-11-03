import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: A_L (2x1) * B_T (1x8)
    # A_L is col 0 of A. B_T is row 0 of B.
    # Terms: i in 0..1, k in 0..7
    # U index: i*m + 0
    # V index: 0*p + k
    # W index: k*n + i
    for i in range(n):
        for k in range(p):
            c = i * p + k
            U[i * m + 0, c] = 1
            V[0 * p + k, c] = 1
            W[k * n + i, c] = 1
            
    # Part 2: A_R (2x2) * B_B (2x8)
    # A_R is cols 1,2 of A. B_B is rows 1,2 of B.
    # Split B_B into 4 blocks of 2 columns each.
    # Block b corresponds to columns 2b, 2b+1 of B (global k).
    # In A_R, local indices 0,1 correspond to global j=1,2.
    # In B_B, local indices 0,1 correspond to global j=1,2.
    
    for b in range(4):
        # Global k indices for this block
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for this 2x2x2 block
        # We will add 7 terms starting at column 16 + 7*b
        base_c = 16 + 7 * b
        
        # M1 = (a_00 + a_11) * (b_00 + b_11)
        # a_00 is A[0,1], a_11 is A[1,2]
        # b_00 is B[1,k0], b_11 is B[2,k1]
        c = base_c + 0
        U[0 * m + 1, c] += 1
        U[1 * m + 2, c] += 1
        V[1 * p + k0, c] += 1
        V[2 * p + k1, c] += 1
        # W contributions:
        # c_00 = M1 ... -> W[k0*n + 0] += 1
        # c_11 = M1 ... -> W[k1*n + 1] += 1
        W[k0 * n + 0, c] += 1
        W[k1 * n + 1, c] += 1
        
        # M2 = (a_10 + a_11) * b_00
        # a_10 is A[1,1], a_11 is A[1,2]
        # b_00 is B[1,k0]
        c = base_c + 1
        U[1 * m + 1, c] += 1
        U[1 * m + 2, c] += 1
        V[1 * p + k0, c] += 1
        # W contributions:
        # c_10 = M2 ... -> W[k0*n + 1] += 1
        W[k0 * n + 1, c] += 1
        
        # M3 = a_00 * (b_10 - b_11)
        # a_00 is A[0,1]
        # b_10 is B[2,k0], b_11 is B[2,k1]
        c = base_c + 2
        U[0 * m + 1, c] += 1
        V[2 * p + k0, c] += 1
        V[2 * p + k1, c] -= 1
        # W contributions:
        # c_01 = M3 ... -> W[k1*n + 0] += 1
        # c_11 = -M3 ... -> W[k1*n + 1] -= 1
        W[k1 * n + 0, c] += 1
        W[k1 * n + 1, c] -= 1
        
        # M4 = a_11 * (b_01 - b_00)
        # a_11 is A[1,2]
        # b_01 is B[1,k1], b_00 is B[1,k0]
        c = base_c + 3
        U[1 * m + 2, c] += 1
        V[1 * p + k1, c] += 1
        V[1 * p + k0, c] -= 1
        # W contributions:
        # c_00 = -M4 ... -> W[k0*n + 0] -= 1
        # c_10 = M4 ... -> W[k0*n + 1] += 1
        W[k0 * n + 0, c] -= 1
        W[k0 * n + 1, c] += 1
        
        # M5 = (a_00 + a_01) * b_11
        # a_00 is A[0,1], a_01 is A[0,2]
        # b_11 is B[2,k1]
        c = base_c + 4
        U[0 * m + 1, c] += 1
        U[0 * m + 2, c] += 1
        V[2 * p + k1, c] += 1
        # W contributions:
        # c_00 = M5 ... -> W[k0*n + 0] += 1
        W[k0 * n + 0, c] += 1
        
        # M6 = (a_10 - a_00) * (b_00 + b_01)
        # a_10 is A[1,1], a_00 is A[0,1]
        # b_00 is B[1,k0], b_01 is B[1,k1]
        c = base_c + 5
        U[1 * m + 1, c] += 1
        U[0 * m + 1, c] -= 1
        V[1 * p + k0, c] += 1
        V[1 * p + k1, c] += 1
        # W contributions:
        # c_11 = M6 ... -> W[k1*n + 1] += 1
        W[k1 * n + 1, c] += 1
        
        # M7 = (a_01 - a_11) * (b_10 + b_11)
        # a_01 is A[0,2], a_11 is A[1,2]
        # b_10 is B[2,k0], b_11 is B[2,k1]
        c = base_c + 6
        U[0 * m + 2, c] += 1
        U[1 * m + 2, c] -= 1
        V[2 * p + k0, c] += 1
        V[2 * p + k1, c] += 1
        # W contributions:
        # c_00 = -M7 ... -> W[k0*n + 0] -= 1
        # c_01 = M7 ... -> W[k1*n + 0] += 1
        W[k0 * n + 0, c] -= 1
        W[k1 * n + 0, c] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
