import numpy as np

def construct():
    """
    Rank-28 exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    
    Decomposition:
    - Block 1: j=0 (2x1x5), rank 10 (schoolbook)
    - Block 2a: j=1,2, k=0,1 (2x2x2 Strassen), rank 7
    - Block 2b: j=1,2, k=2 (2x2x1 schoolbook), rank 4
    - Block 2c: j=1,2, k=3,4 (2x2x2 Strassen), rank 7
    Total: 28
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Block 1: j=0, all k (2x1x5 schoolbook, rank 10)
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        U indices: (i0+i)*m + (j0+j)
        V indices: (j0+j)*p + (k0+k)
        W indices: (k0+k)*n + (i0+i)
        """
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[(i0+0)*m + (j0+0), col] = 1; U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+0)*p + (k0+0), col] = 1; V[(j0+1)*p + (k0+1), col] = 1
        W[(k0+0)*n + (i0+0), col] = 1; W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        # M2: (A10+A11)B00 -> C10, -C11
        U[(i0+1)*m + (j0+0), col] = 1; U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+0)*p + (k0+0), col] = 1
        W[(k0+0)*n + (i0+1), col] = 1; W[(k0+1)*n + (i0+1), col] = -1
        col += 1
        # M3: A00(B01-B11) -> C01, C11
        U[(i0+0)*m + (j0+0), col] = 1
        V[(j0+0)*p + (k0+1), col] = 1; V[(j0+1)*p + (k0+1), col] = -1
        W[(k0+1)*n + (i0+0), col] = 1; W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        # M4: A11(B10-B00) -> C00, C10
        U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+1)*p + (k0+0), col] = 1; V[(j0+0)*p + (k0+0), col] = -1
        W[(k0+0)*n + (i0+0), col] = 1; W[(k0+0)*n + (i0+1), col] = 1
        col += 1
        # M5: (A00+A01)B11 -> -C00, C01
        U[(i0+0)*m + (j0+0), col] = 1; U[(i0+0)*m + (j0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[(k0+0)*n + (i0+0), col] = -1; W[(k0+1)*n + (i0+0), col] = 1
        col += 1
        # M6: (A10-A00)(B00+B01) -> C11
        U[(i0+1)*m + (j0+0), col] = 1; U[(i0+0)*m + (j0+0), col] = -1
        V[(j0+0)*p + (k0+0), col] = 1; V[(j0+0)*p + (k0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        # M7: (A01-A11)(B10+B11) -> C00
        U[(i0+0)*m + (j0+1), col] = 1; U[(i0+1)*m + (j0+1), col] = -1
        V[(j0+1)*p + (k0+0), col] = 1; V[(j0+1)*p + (k0+1), col] = 1
        W[(k0+0)*n + (i0+0), col] = 1
        col += 1
    
    # Block 2a: j=1,2, k=0,1 (2x2x2 Strassen, rank 7)
    add_strassen(0, 1, 0)
    
    # Block 2b: j=1,2, k=2 (2x2x1 schoolbook, rank 4)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (2x2x2 Strassen, rank 7)
    add_strassen(0, 1, 3)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
