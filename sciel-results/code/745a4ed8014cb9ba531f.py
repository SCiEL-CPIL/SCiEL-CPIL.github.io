"""Matrix multiplication 2x2 x 2x16 via Strassen decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses Strassen's rank-7 decomposition for the 2x2 core, applied
    independently to each of the 8 column-pairs of the 2x16 matrix.
    Total rank = 7 * 8 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen intermediates for 2x2 multiplication
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A (U):
    # A00: 0, A01: 1, A10: 2, A11: 3
    
    # For each block t (0..7), B columns are 2t, 2t+1
    # B00 index in V: 0*p + 2t = 2t
    # B01 index in V: 0*p + 2t+1 = 2t+1
    # B10 index in V: 1*p + 2t = 16+2t
    # B11 index in V: 1*p + 2t+1 = 16+2t+1
    
    # For each block t, C columns are 2t, 2t+1
    # C00 index in W: 2t*n + 0 = 4t
    # C01 index in W: (2t+1)*n + 0 = 4t+2
    # C10 index in W: 2t*n + 1 = 4t+1
    # C11 index in W: (2t+1)*n + 1 = 4t+3
    
    # Note: The problem statement says W[k*n+i, r].
    # So for column k, row i, index is k*2 + i.
    # C00 (i=0, k=2t) -> 2t*2 + 0 = 4t
    # C10 (i=1, k=2t) -> 2t*2 + 1 = 4t+1
    # C01 (i=0, k=2t+1) -> (2t+1)*2 + 0 = 4t+2
    # C11 (i=1, k=2t+1) -> (2t+1)*2 + 1 = 4t+3
    
    col = 0
    for t in range(num_blocks):
        if _time_remaining() < 1.0:
            break
            
        b00 = 2*t
        b01 = 2*t + 1
        b10 = 16 + 2*t
        b11 = 16 + 2*t + 1
        
        c00 = 4*t
        c10 = 4*t + 1
        c01 = 4*t + 2
        c11 = 4*t + 3
        
        # M1
        U[0, col] += 1; U[3, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # M2
        U[2, col] += 1; U[3, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        
        # M3
        U[0, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # M4
        U[3, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # M5
        U[0, col] += 1; U[1, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        
        # M6
        U[2, col] += 1; U[0, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7
        U[1, col] += 1; U[3, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
