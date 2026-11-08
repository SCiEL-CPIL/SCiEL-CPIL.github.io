import numpy as np
import time

_START = time.time()
_DEADLINE = 130.0

def construct():
    n, m, p = 4, 4, 7
    
    # Laderman's algorithm for 4x4 x 4x4, using 2x2 block structure.
    # A = [[A11, A12], [A21, A22]], each block 2x2
    # B = [[B11, B12], [B21, B22]]
    
    # Element mapping:
    # A11 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]  -> flat indices 0,1,4,5
    # A12 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]  -> flat indices 2,3,6,7
    # A21 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]  -> flat indices 8,9,12,13
    # A22 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]  -> flat indices 10,11,14,15
    
    # The 13 products (in terms of 2x2 matrix multiplications):
    # P1 = A11 * B11
    # P2 = (A11+A12) * B21
    # P3 = A21 * (B11+B12)
    # P4 = (A22-A21) * (B21+B22)
    # P5 = (A11+A22) * (B11+B22)
    # P6 = (A12+A22) * (B22-B11)
    # P7 = (A21+A11) * (B12-B22)
    # P8 = (A12-A11+A21) * B11
    # P9 = (A11+A21-A22) * B22
    # P10 = (A12+A22-A21) * (B11+B12)
    # P11 = (A11-A22+A21) * (B21+B22)
    # P12 = (A22-A11+A12) * (B22-B12)
    # P13 = (A21-A12+A22) * (B21-B11)
    
    # Output:
    # C11 = P1 + P4 - P5 + P8
    # C12 = P6 + P7 + P10 - P13
    # C21 = P2 + P3 - P9 + P11
    # C22 = P5 + P9 - P12 + P13
    
    # Wait, I need to verify these output formulas. Let me check with a simple case.
    # If A = I (identity) and B = B, then C = B.
    # A11 = I2, A12 = 0, A21 = 0, A22 = I2
    # P1 = I2 * B11 = B11
    # P2 = I2 * B21 = B21
    # P3 = 0 * (B11+B12) = 0
    # P4 = (I2-0) * (B21+B22) = B21+B22
    # P5 = (I2+I2) * (B11+B22) = 2*(B11+B22)
    # P6 = (0+I2) * (B22-B11) = B22-B11
    # P7 = (0+I2) * (B12-B22) = B12-B22
    # P8 = (0-I2+0) * B11 = -B11
    # P9 = (I2+0-I2) * B22 = 0
    # P10 = (0+I2-0) * (B11+B12) = B11+B12
    # P11 = (I2-I2+0) * (B21+B22) = 0
    # P12 = (I2-I2+0) * (B22-B12) = 0
    # P13 = (0-0+I2) * (B21-B11) = B21-B11
    
    # C11 should be B11: P1 + P4 - P5 + P8 = B11 + (B21+B22) - 2*(B11+B22) + (-B11)
    #    = B11 + B21 + B22 - 2*B11 - 2*B22 - B11 = -B11 + B21 - B22
    # That's not B11! So my output formulas are wrong.
    
    # Let me look up the correct Laderman output formulas.
    # 
    # After checking, the correct output is:
    # C11 = P1 + P4 - P5 + P8  ... this gives wrong answer for A=I
    # 
    # I think the issue is that I have the product definitions or output formulas wrong.
    
    # Let me abandon this approach and just use schoolbook. The risk of getting 
    # Laderman wrong is too high without being able to verify interactively.
    
    # Schoolbook rank 112:
    rank = 112
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    for i in range(4):
        for j in range(4):
            for k in range(7):
                r = i * 28 + j * 7 + k
                U[i * 4 + j, r] = 1.0
                V[j * 7 + k, r] = 1.0
                W[k * 4 + i, r] = 1.0
    
    U_out = [[float(U[i, r]) for r in range(rank)] for i in range(16)]
    V_out = [[float(V[i, r]) for r in range(rank)] for i in range(28)]
    W_out = [[float(W[i, r]) for r in range(rank)] for i in range(28)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
