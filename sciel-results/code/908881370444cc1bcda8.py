"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-39 exact tensor decomposition for 2x2 by 2x11 matrix multiplication
    using Strassen's algorithm for 2x2x2 blocks and schoolbook for the remaining 2x2x1 block.
    """
    n, m, p = DIMENSIONS
    # Total rank: 5 blocks of 7 + 1 block of 4 = 39
    rank = 39
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Define Strassen factors for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # We process 5 blocks of 2 columns each
    for block_idx in range(5):
        k_start = block_idx * 2
        
        # Term 1: M1
        U[0, col] = 1; U[3, col] = 1 # A00, A11
        V[0 + k_start, col] = 1; V[2 + k_start, col] = 1 # B00, B11 (j=0,1 local -> global j=0,1)
        # Note: V index is j*p + k.
        # Local B is 2x2. Rows j=0,1. Cols k'=0,1.
        # Global k = k_start + k'.
        # V[j*p + k].
        # B00 is j=0, k=k_start. V[0*p + k_start]
        # B11 is j=1, k=k_start+1. V[1*p + k_start+1]
        # Wait, my previous manual trace was:
        # B00 -> idx 0 (if k_start=0)
        # B11 -> idx 12 (if k_start=0, j=1, k=1 -> 1*11+1=12)
        # Let's re-verify V index formula: j*p + k.
        # p=11.
        # Block 0: k_start=0.
        # B00: j=0, k=0 -> V[0*11+0] = V[0]
        # B11: j=1, k=1 -> V[1*11+1] = V[12]
        
        # Correcting V assignments for Strassen M1:
        # (B00 + B11) means V has 1 at V[0] and V[12] (for block 0).
        
        # Let's generalize for block_idx:
        # k0 = k_start
        # k1 = k_start + 1
        # B00: j=0, k=k0 -> V[0*p + k0]
        # B01: j=0, k=k1 -> V[0*p + k1]
        # B10: j=1, k=k0 -> V[1*p + k0]
        # B11: j=1, k=k1 -> V[1*p + k1]
        
        # M1: (A00+A11)(B00+B11)
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k_start, col] = 1
        V[1*p + k_start+1, col] = 1
        # W: C00 + C11
        # C00: i=0, k=k0 -> W[k0*n + 0]
        # C11: i=1, k=k1 -> W[k1*n + 1]
        W[k_start*n + 0, col] = 1
        W[(k_start+1)*n + 1, col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[2, col] = 1; U[3, col] = 1 # A10 (i=1,j=0 -> idx 2), A11 (i=1,j=1 -> idx 3)
        V[0*p + k_start, col] = 1 # B00
        # W: C10 - C11
        # C10: i=1, k=k0 -> W[k0*n + 1]
        # C11: i=1, k=k1 -> W[k1*n + 1]
        W[k_start*n + 1, col] = 1
        W[(k_start+1)*n + 1, col] = -1
        col += 1
        
        # M3: A00(B01-B11)
        U[0, col] = 1 # A00
        V[0*p + k_start+1, col] = 1 # B01
        V[1*p + k_start+1, col] = -1 # B11
        # W: C01
        # C01: i=0, k=k1 -> W[k1*n + 0]
        W[(k_start+1)*n + 0, col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        U[3, col] = 1 # A11
        V[1*p + k_start, col] = 1 # B10
        V[0*p + k_start, col] = -1 # B00
        # W: C00 + C10
        # C00: i=0, k=k0 -> W[k0*n + 0]
        # C10: i=1, k=k0 -> W[k0*n + 1]
        W[k_start*n + 0, col] = 1
        W[k_start*n + 1, col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        U[0, col] = 1; U[1, col] = 1 # A00, A01
        V[1*p + k_start+1, col] = 1 # B11
        # W: -C00 + C01
        # C00: i=0, k=k0 -> W[k0*n + 0]
        # C01: i=0, k=k1 -> W[k1*n + 0]
        W[k_start*n + 0, col] = -1
        W[(k_start+1)*n + 0, col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[2, col] = 1; U[0, col] = -1 # A10, A00
        V[0*p + k_start, col] = 1 # B00
        V[0*p + k_start+1, col] = 1 # B01
        # W: C11
        # C11: i=1, k=k1 -> W[k1*n + 1]
        W[(k_start+1)*n + 1, col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[1, col] = 1; U[3, col] = -1 # A01, A11
        V[1*p + k_start, col] = 1 # B10
        V[1*p + k_start+1, col] = 1 # B11
        # W: C00
        # C00: i=0, k=k0 -> W[k0*n + 0]
        W[k_start*n + 0, col] = 1
        col += 1

    # Last block: column 10 (k_start = 10)
    # Schoolbook 2x2x1
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    
    # Term 1: A00 B00 -> C00
    U[0, col] = 1
    V[0*p + 10, col] = 1
    W[10*n + 0, col] = 1
    col += 1
    
    # Term 2: A01 B10 -> C00
    U[1, col] = 1
    V[1*p + 10, col] = 1
    W[10*n + 0, col] = 1
    col += 1
    
    # Term 3: A10 B00 -> C10
    U[2, col] = 1
    V[0*p + 10, col] = 1
    W[10*n + 1, col] = 1
    col += 1
    
    # Term 4: A11 B10 -> C10
    U[3, col] = 1
    V[1*p + 10, col] = 1
    W[10*n + 1, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
