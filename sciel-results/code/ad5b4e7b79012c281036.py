"""Strassen's rank-7 decomposition for 2x2 matrix multiplication."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm: rank-7 decomposition for 2x2 matrix multiplication.
    # U[i*m+j, r] extracts linear forms from A
    # V[j*p+k, r] extracts linear forms from B
    # W[k*n+i, r] = C[i,k] (official trace convention)
    
    # U: rows indexed by i*2+j (A[i,j]), 7 columns for 7 products
    U = [
        [1, 0, 1, 0, 1, 0, 1],   # A[0,0]: P1, P3, P5, P7
        [0, 0, 0, 0, 0, -1, -1], # A[0,1]: P6, P7
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]: P2, P4, P6
        [1, 1, 0, 1, 0, 0, 0],   # A[1,1]: P1, P2, P4
    ]
    
    # V: rows indexed by j*2+k (B[j,k]), 7 columns
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]: P1, P2, P4, P6
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]: P3, P6
        [1, 0, -1, 0, 1, 0, 1],  # B[1,0]: P1, P3, P5, P7
        [1, 0, 0, 0, 1, 0, 1],   # B[1,1]: P1, P5, P7
    ]
    
    # W: rows indexed by k*2+i (C[i,k]), 7 columns
    # W[k*n+i, r] where n=2, so W[k*2+i, r]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0] = P1 + P4 - P5 + P7
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0] = P2 + P4
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1] = P3 + P5
        [1, -1, 1, 0, 0, 1, 0],  # C[1,1] = P1 - P2 + P3 + P6
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
