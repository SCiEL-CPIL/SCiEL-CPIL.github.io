# EVOLVE-BLOCK-START
def construct():
    """
    Rank-7 tensor decomposition for 2x2x2 matrix multiplication (Strassen).
    
    Convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = 7
    
    # U: 4x7 (indices: 0=a, 1=b, 2=c, 3=d)
    U = [
        [1, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 1, 0]
    ]
    
    # V: 4x7 (indices: 0=e, 1=f, 2=g, 3=h)
    V = [
        [1, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, -1, -1, 0, -1, 1]
    ]
    
    # W: 4x7 (indices: 0=C00, 1=C10, 2=C01, 3=C11)
    # Note: W[k*n+i] means:
    # C00 (i=0,k=0) -> idx 0
    # C10 (i=1,k=0) -> idx 1
    # C01 (i=0,k=1) -> idx 2
    # C11 (i=1,k=1) -> idx 3
    W = [
        [1, 0, 0, 1, 1, 0, -1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0]
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
