"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 3, 10
    
    # Rank 55 decomposition
    # 5 blocks of (2x3x2) subproblems
    # Each 2x3x2 is decomposed into:
    # - 2x2x2 part (cols 0,1 of A, rows 0,1 of B) using Strassen (Rank 7)
    # - 2x1x2 part (col 2 of A, row 2 of B) using Schoolbook (Rank 4)
    # Total per block: 11
    # Total: 5 * 11 = 55
    
    rank = 55
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    
    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00      -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)      -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)      -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11      -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01)-> C11 += M6
    # M7 = (A01-A11)(B10+B11)-> C00 += M7
    
    # Indices for 2x2x2 block
    # A indices: 0,1 (row 0), 3,4 (row 1) -> i*m+j
    # B indices: 0,1 (row 0), 10,11 (row 1) -> j*p+k
    # C indices: 0,1 (col 0), 2,3 (col 1) -> k*n+i
    
    # We iterate over 5 blocks of 2 columns in B
    for block in range(5):
        k_start = 2 * block
        k_end = k_start + 2
        
        # B indices for this block
        # Row 0: k_start, k_start+1
        # Row 1: 10+k_start, 10+k_start+1
        b00 = k_start
        b01 = k_start + 1
        b10 = 10 + k_start
        b11 = 10 + k_start + 1
        
        # C indices for this block
        # k=k_start: 0, 1 (i=0,1)
        # k=k_start+1: 2, 3 (i=0,1)
        c00 = k_start * 2 + 0
        c01 = k_start * 2 + 1
        c10 = (k_start + 1) * 2 + 0
        c11 = (k_start + 1) * 2 + 1
        
        # Strassen Terms
        # 1. M1
        U[0, idx] += 1; U[4, idx] += 1
        V[b00, idx] += 1; V[b11, idx] += 1
        W[c00, idx] += 1; W[c11, idx] += 1
        idx += 1
        
        # 2. M2
        U[3, idx] += 1; U[4, idx] += 1
        V[b00, idx] += 1
        W[c10, idx] += 1; W[c11, idx] -= 1
        idx += 1
        
        # 3. M3
        U[0, idx] += 1
        V[b01, idx] += 1; V[b11, idx] -= 1
        W[c01, idx] += 1; W[c11, idx] += 1
        idx += 1
        
        # 4. M4
        U[4, idx] += 1
        V[b10, idx] += 1; V[b00, idx] -= 1
        W[c00, idx] += 1; W[c10, idx] += 1
        idx += 1
        
        # 5. M5
        U[0, idx] += 1; U[1, idx] += 1
        V[b11, idx] += 1
        W[c00, idx] -= 1; W[c01, idx] += 1
        idx += 1
        
        # 6. M6
        U[3, idx] += 1; U[0, idx] -= 1
        V[b00, idx] += 1; V[b01, idx] += 1
        W[c11, idx] += 1
        idx += 1
        
        # 7. M7
        U[1, idx] += 1; U[4, idx] -= 1
        V[b10, idx] += 1; V[b11, idx] += 1
        W[c00, idx] += 1
        idx += 1
        
        # Schoolbook Terms for j=2
        # A indices: 2 (i=0), 5 (i=1)
        # B indices: 20+k_start, 20+k_start+1
        # C indices: same as above
        
        a02 = 2
        a12 = 5
        b20 = 20 + k_start
        b21 = 20 + k_start + 1
        
        # 8. A02 * B20 -> C00
        U[a02, idx] += 1
        V[b20, idx] += 1
        W[c00, idx] += 1
        idx += 1
        
        # 9. A02 * B21 -> C01
        U[a02, idx] += 1
        V[b21, idx] += 1
        W[c01, idx] += 1
        idx += 1
        
        # 10. A12 * B20 -> C10
        U[a12, idx] += 1
        V[b20, idx] += 1
        W[c10, idx] += 1
        idx += 1
        
        # 11. A12 * B21 -> C11
        U[a12, idx] += 1
        V[b21, idx] += 1
        W[c11, idx] += 1
        idx += 1

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
