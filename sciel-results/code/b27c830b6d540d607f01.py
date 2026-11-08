import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2 rows of A, m=3 intermediate, p=10 columns of B
    
    # Split the intermediate dimension m=3 into m1=2 and m2=1
    m1 = 2
    m2 = 1
    
    # Split p=10 into pairs: 5 pairs of 2 columns
    # For Strassen on 2x2x2 blocks
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    idx = 0
    
    # Part 1: A_left (2x2) @ B_top (2x10) using Strassen per column pair
    # A_left has indices: row i in [0,1], col j in [0,1] -> U index i*m + j
    # B_top has indices: row j in [0,1], col k in [0,9] -> V index j*p + k
    # Output C[i,k] -> W index k*n + i
    
    # Strassen's 7 products for 2x2 @ 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A10)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    for col_pair in range(5):
        k0 = col_pair * 2
        k1 = col_pair * 2 + 1
        
        # M1 = (A00+A11)(B00+B11)
        # A00: i=0,j=0 -> U[0*3+0] = U[0]
        # A11: i=1,j=1 -> U[1*3+1] = U[4]
        # B00: j=0,k=k0 -> V[0*10+k0] = V[k0]
        # B11: j=1,k=k1 -> V[1*10+k1] = V[10+k1]
        # Output: C00 += M1, C11 += M1 -> W[k0*n+0]=W[k0*2+0], W[k1*n+1]=W[k1*2+1]
        U[0, idx] = 1
        U[4, idx] = 1
        V[k0, idx] = 1
        V[10 + k1, idx] = 1
        W[k0 * 2 + 0, idx] = 1  # C[0, k0]
        W[k1 * 2 + 1, idx] = 1  # C[1, k1]
        idx += 1
        
        # M2 = (A10+A11)B00
        # A10: i=1,j=0 -> U[1*3+0] = U[3]
        # A11: i=1,j=1 -> U[4]
        # B00: j=0,k=k0 -> V[k0]
        # Output: C10 += M2, C11 -= M2 -> W[k0*2+1]=W[k0*2+1], W[k1*2+1]=W[k1*2+1]
        U[3, idx] = 1
        U[4, idx] = 1
        V[k0, idx] = 1
        W[k0 * 2 + 1, idx] = 1   # C[1, k0]
        W[k1 * 2 + 1, idx] = -1  # C[1, k1]
        idx += 1
        
        # M3 = A00(B01-B11)
        # A00: i=0,j=0 -> U[0]
        # B01: j=0,k=k1 -> V[0*10+k1] = V[k1]
        # B11: j=1,k=k1 -> V[10+k1]
        # Output: C01 += M3, C11 += M3 -> W[k1*2+0]=W[k1*2+0], W[k1*2+1]=W[k1*2+1]
        U[0, idx] = 1
        V[k1, idx] = 1
        V[10 + k1, idx] = -1
        W[k1 * 2 + 0, idx] = 1   # C[0, k1]
        W[k1 * 2 + 1, idx] = 1   # C[1, k1]
        idx += 1
        
        # M4 = A11(B10-B00)
        # A11: i=1,j=1 -> U[4]
        # B10: j=1,k=k0 -> V[10+k0]
        # B00: j=0,k=k0 -> V[k0]
        # Output: C00 += M4, C10 += M4 -> W[k0*2+0]=W[k0*2+0], W[k0*2+1]=W[k0*2+1]
        U[4, idx] = 1
        V[10 + k0, idx] = 1
        V[k0, idx] = -1
        W[k0 * 2 + 0, idx] = 1   # C[0, k0]
        W[k0 * 2 + 1, idx] = 1   # C[1, k0]
        idx += 1
        
        # M5 = (A00+A01)B11
        # A00: i=0,j=0 -> U[0]
        # A01: i=0,j=1 -> U[0*3+1] = U[1]
        # B11: j=1,k=k1 -> V[10+k1]
        # Output: C00 -= M5, C01 += M5 -> W[k0*2+0]=W[k0*2+0], W[k1*2+0]=W[k1*2+0]
        U[0, idx] = 1
        U[1, idx] = 1
        V[10 + k1, idx] = 1
        W[k0 * 2 + 0, idx] = -1  # C[0, k0]
        W[k1 * 2 + 0, idx] = 1   # C[0, k1]
        idx += 1
        
        # M6 = (A10-A00)(B00+B01)
        # A10: i=1,j=0 -> U[3]
        # A00: i=0,j=0 -> U[0]
        # B00: j=0,k=k0 -> V[k0]
        # B01: j=0,k=k1 -> V[k1]
        # Output: C11 += M6 -> W[k1*2+1]=W[k1*2+1]
        U[3, idx] = 1
        U[0, idx] = -1
        V[k0, idx] = 1
        V[k1, idx] = 1
        W[k1 * 2 + 1, idx] = 1   # C[1, k1]
        idx += 1
        
        # M7 = (A01-A10)(B10+B11)
        # A01: i=0,j=1 -> U[1]
        # A10: i=1,j=0 -> U[3]
        # B10: j=1,k=k0 -> V[10+k0]
        # B11: j=1,k=k1 -> V[10+k1]
        # Output: C00 += M7 -> W[k0*2+0]=W[k0*2+0]
        U[1, idx] = 1
        U[3, idx] = -1
        V[10 + k0, idx] = 1
        V[10 + k1, idx] = 1
        W[k0 * 2 + 0, idx] = 1   # C[0, k0]
        idx += 1
    
    # Part 2: A_right (2x1) @ B_bottom (1x10)
    # A_right: row i in [0,1], col j=2 -> U[i*3+2] = U[2], U[5]
    # B_bottom: row j=2, col k in [0,9] -> V[2*10+k] = V[20+k]
    # Output: C[i,k] += A_right[i] * B_bottom[k] -> W[k*n+i] = W[k*2+i]
    
    for i in range(2):
        for k in range(10):
            U[i * m + m1, idx] = 1  # U[i*3+2]
            V[m1 * p + k, idx] = 1  # V[2*10+k] = V[20+k]
            W[k * n + i, idx] = 1   # W[k*2+i]
            idx += 1
    
    assert idx == rank, f"Expected rank {rank}, got {idx}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
