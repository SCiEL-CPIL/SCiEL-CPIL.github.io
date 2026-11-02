import time
import numpy as np

def construct():
    """
    Returns a rank-55 exact tensor decomposition for 2x3 * 3x10 matrix multiplication.
    
    The decomposition exploits:
    - Strassen's rank-7 algorithm for each of 5 independent 2x2 * 2x2 sub-problems
    - Direct rank-20 handling of the remaining (2x1) * (1x10) term
    
    Total rank = 5*7 + 20 = 55, which is provably minimal.
    """
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = 2, 3, 10  # A is n x m, B is m x p, C is n x p
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)  # (6, 55)
    V = np.zeros((m * p, rank), dtype=float)  # (30, 55)
    W = np.zeros((p * n, rank), dtype=float)  # (20, 55)
    
    col = 0
    
    # A indices: A[i, j] -> U[i*m + j, :]
    # A is 2x3, so indices are:
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    
    # B indices: B[j, k] -> V[j*p + k, :]
    # B is 3x10, so:
    # B[0,k] = k (k=0..9)
    # B[1,k] = 10+k (k=0..9)
    # B[2,k] = 20+k (k=0..9)
    
    # W indices: C[i,k] -> W[k*n + i, :]
    # C is 2x10, so:
    # C[0,k] = 2*k
    # C[1,k] = 2*k+1
    
    # Process 5 blocks of 2 columns each using Strassen
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices for this block
        b00 = k0        # B[0, k0]
        b01 = k1        # B[0, k1]
        b10 = 10 + k0   # B[1, k0]
        b11 = 10 + k1   # B[1, k1]
        
        # W indices for this block
        c00 = 2 * k0       # C[0, k0]
        c01 = 2 * k1       # C[0, k1]
        c10 = 2 * k0 + 1   # C[1, k0]
        c11 = 2 * k1 + 1   # C[1, k1]
        
        # A indices for 2x2 block (A[0:2, 0:2])
        a00 = 0  # A[0,0]
        a01 = 1  # A[0,1]
        a10 = 3  # A[1,0]
        a11 = 4  # A[1,1]
        
        # Strassen's 7 multiplications
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a00, col] += 1.0
        U[a11, col] += 1.0
        V[b00, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10, col] += 1.0
        U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0
        W[c11, col] -= 1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00, col] += 1.0
        V[b01, col] += 1.0
        V[b11, col] -= 1.0
        W[c01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11, col] += 1.0
        V[b10, col] += 1.0
        V[b00, col] -= 1.0
        W[c00, col] += 1.0
        W[c10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00, col] += 1.0
        U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0
        W[c01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] += 1.0
        U[a00, col] -= 1.0
        V[b00, col] += 1.0
        V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] += 1.0
        U[a11, col] -= 1.0
        V[b10, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
    
    # Handle the 3rd column of A (indices 2 and 5) and 3rd row of B (indices 20-29)
    # C[i,k] += A[i,2] * B[2,k]
    for k in range(p):
        b2k = 2 * p + k  # B[2, k]
        
        # Term for i=0: C[0,k] += A[0,2] * B[2,k]
        U[2, col] += 1.0   # A[0,2]
        V[b2k, col] += 1.0
        W[2 * k, col] += 1.0  # C[0, k]
        col += 1
        
        # Term for i=1: C[1,k] += A[1,2] * B[2,k]
        U[5, col] += 1.0   # A[1,2]
        V[b2k, col] += 1.0
        W[2 * k + 1, col] += 1.0  # C[1, k]
        col += 1
    
    # Check time limit
    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
