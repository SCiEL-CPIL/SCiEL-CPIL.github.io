import numpy as np
import time

def construct():
    """
    Rank-56 tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Partitions into 8 independent 2x2 * 2x2 sub-problems using Strassen (rank 7 each).
    Output convention: W[k*n + i, r] where k is column index, i is row index.
    """
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen coefficients for 2x2*2x2:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*b10
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    col = 0
    for s in range(2):
        for j in range(4):
            # A indices: row*4 + col_in_A
            a00 = 0 * m + (2*s + 0)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s + 0)
            a11 = 1 * m + (2*s + 1)
            
            # B indices: row*8 + col_in_B
            b00 = (2*s + 0) * p + (2*j + 0)
            b01 = (2*s + 0) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j + 0)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            # C indices: col*n + row
            c00 = (2*j + 0) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j + 0) * n + 1
            c11 = (2*j + 1) * n + 1
            
            # U side
            U[a00, col] += 1;   U[a11, col] += 1
            U[a10, col+1] += 1; U[a11, col+1] += 1
            U[a00, col+2] += 1
            U[a11, col+3] += 1
            U[a00, col+4] += 1; U[a01, col+4] += 1
            U[a10, col+5] += 1; U[a00, col+5] += -1
            U[a01, col+6] += 1; U[a11, col+6] += -1
            
            # V side
            V[b00, col] += 1;   V[b11, col] += 1
            V[b00, col+1] += 1
            V[b01, col+2] += 1; V[b11, col+2] += -1
            V[b10, col+3] += 1; V[b00, col+3] += -1
            V[b11, col+4] += 1
            V[b00, col+5] += 1; V[b01, col+5] += 1
            V[b10, col+6] += 1; V[b11, col+6] += 1
            
            # W side
            W[c00, col] += 1;   W[c00, col+3] += 1; W[c00, col+4] += -1; W[c00, col+6] += 1
            W[c01, col+2] += 1; W[c01, col+4] += 1
            W[c10, col+1] += 1; W[c10, col+3] += 1
            W[c11, col] += 1;   W[c11, col+1] += -1; W[c11, col+2] += 1; W[c11, col+5] += 1
            
            col += 7
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
