# EVOLVE-BLOCK-START
import numpy as np

def _strassen_rank7():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U: (4, 7) for A (flattened row-major: A00, A01, A10, A11)
    V: (4, 7) for B (flattened row-major: B00, B01, B10, B11)
    W: (4, 7) for C (flattened column-major index k*n+i: C00, C10, C01, C11)
    
    Note on W indexing:
    The task specifies W[k*n+i, r].
    For n=2, p=2:
    Index 0: k=0, i=0 -> C[0,0]
    Index 1: k=0, i=1 -> C[1,0]
    Index 2: k=1, i=0 -> C[0,1]
    Index 3: k=1, i=1 -> C[1,1]
    """
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (k=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]   # C11 (k=1, i=1)
    ]
    
    return (np.array(U_data, dtype=float),
            np.array(V_data, dtype=float),
            np.array(W_data, dtype=float))

def construct():
    """
    Returns the rank-minimal (rank 7) exact tensor decomposition for 2x2 matrix multiplication.
    """
    U, V, W = _strassen_rank7()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
