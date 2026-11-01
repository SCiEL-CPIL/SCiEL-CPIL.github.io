# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    
    rank = 7
    
    # U shape (n*m, rank) -> (4, 7)
    # Row index i*m + j
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]
    
    # V shape (m*p, rank) -> (4, 7)
    # Row index j*p + k
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]
    
    # W shape (p*n, rank) -> (4, 7)
    # Row index k*n + i
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0] (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0] (k=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1] (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1] (k=1, i=1)
    ]
    
    U = np.array(U_data, dtype=int)
    V = np.array(V_data, dtype=int)
    W = np.array(W_data, dtype=int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
