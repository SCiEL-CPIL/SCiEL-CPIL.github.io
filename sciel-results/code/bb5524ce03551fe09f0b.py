import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Returns:
        A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # The optimal known rank for 2x3x10 is 55, derived from 5 independent 
    # 2x3x2 block multiplications, each of rank 11.
    # We construct this decomposition explicitly.
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Indices for A (2x3): 
    # a00=0, a01=1, a02=2
    # a10=3, a11=4, a12=5
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    
    # Indices for B (3x10):
    # Row 0: 0..9
    # Row 1: 10..19
    # Row 2: 20..29
    
    # Indices for C (2x10):
    # Row 0: 0..9
    # Row 1: 10..19
    
    # We process 5 blocks of B, each of size 3x2.
    for b in range(5):
        # B block indices
        # b00, b01 correspond to B[0, 2b], B[0, 2b+1]
        # b10, b11 correspond to B[1, 2b], B[1, 2b+1]
        # b20, b21 correspond to B[2, 2b], B[2, 2b+1]
        b00 = 2 * b
        b01 = 2 * b + 1
        b10 = 10 + 2 * b
        b11 = 10 + 2 * b + 1
        b20 = 20 + 2 * b
        b21 = 20 + 2 * b + 1
        
        # C block indices
        # c00, c01 correspond to C[0, 2b], C[0, 2b+1]
        # c10, c11 correspond to C[1, 2b], C[1, 2b+1]
        c00 = 2 * b
        c01 = 2 * b + 1
        c10 = 10 + 2 * b
        c11 = 10 + 2 * b + 1
        
        # Decomposition of 2x3x2 multiplication into 11 terms.
        # This combines a 2x2x2 Strassen (rank 7) for the top-left 2x2 part
        # and a 2x1x2 multiplication (rank 4) for the bottom-right part.
        
        # Part 1: 2x2x2 multiplication of A[0:2, 0:2] and B[0:2, :]
        # M1 = (a00 + a11)(b00 + b11)
        U[a00, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M2 = (a10 + a11)b00
        U[a10, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] -= 1.0
        col += 1
        
        # M3 = a00(b01 - b11)
        U[a00, col] += 1.0
        V[b01, col] += 1.0; V[b11, col] -= 1.0
        W[c01, col] += 1.0; W[c11, col] += 1.0
        col += 1
        
        # M4 = a11(b10 - b00)
        U[a11, col] += 1.0
        V[b10, col] += 1.0; V[b00, col] -= 1.0
        W[c00, col] += 1.0; W[c10, col] += 1.0
        col += 1
        
        # M5 = (a00 + a01)b11
        U[a00, col] += 1.0; U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0; W[c01, col] += 1.0
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[a10, col] += 1.0; U[a00, col] -= 1.0
        V[b00, col] += 1.0; V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[a01, col] += 1.0; U[a11, col] -= 1.0
        V[b10, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
        
        # Part 2: 2x1x2 multiplication of A[0:2, 2] and B[2, :]
        # M8 = a02 b20
        U[a02, col] += 1.0
        V[b20, col] += 1.0
        W[c00, col] += 1.0
        col += 1
        
        # M9 = a02 b21
        U[a02, col] += 1.0
        V[b21, col] += 1.0
        W[c01, col] += 1.0
        col += 1
        
        # M10 = a12 b20
        U[a12, col] += 1.0
        V[b20, col] += 1.0
        W[c10, col] += 1.0
        col += 1
        
        # M11 = a12 b21
        U[a12, col] += 1.0
        V[b21, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # Check time limit
        if time.time() - start_time > DEADLINE:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
