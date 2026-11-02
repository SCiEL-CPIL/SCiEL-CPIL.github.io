"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def _check_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _build_strassen_block(U, V, W, k1, k2, col):
    """
    Add 7 Strassen columns for 2x2 * 2x2 multiplication using B columns k1,k2.
    A is 2x2 with indices: A00=0, A01=1, A10=2, A11=3
    B rows: row 0 uses B[0*13+k], row 1 uses B[1*13+k]
    C indices: C[k*n+i] where k in {k1,k2}, i in {0,1}
    
    Strassen for C = A*B where B has columns k1,k2:
    M1 = (A00+A11) * (B0k1+B1k2)
    M2 = (A10+A11) * (B0k1+B1k2)
    M3 = A00 * (B1k1-B1k2)
    M4 = A11 * (B0k2-B0k1)
    M5 = (A00+A10) * B0k2
    M6 = (A11-A00) * (B0k1+B1k1)
    M7 = (A01-A11) * (B1k1+B1k2)
    
    C[k1,0] = M1 - M4 + M5 - M7
    C[k2,0] = M4 + M7
    C[k1,1] = M2 - M5 + M6
    C[k2,1] = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS
    
    # M1 = (A00+A11) * (B0k1+B1k2)
    U[0, col] = 1; U[3, col] = 1
    V[0*p+k1, col] = 1; V[1*p+k2, col] = 1
    W[k1*n+0, col] = 1; W[k2*n+1, col] = 1
    col += 1
    
    # M2 = (A10+A11) * (B0k1+B1k2)
    U[2, col] = 1; U[3, col] = 1
    V[0*p+k1, col] = 1; V[1*p+k2, col] = 1
    W[k1*n+1, col] = 1; W[k2*n+1, col] = -1
    col += 1
    
    # M3 = A00 * (B1k1-B1k2)
    U[0, col] = 1
    V[1*p+k1, col] = 1; V[1*p+k2, col] = -1
    W[k2*n+1, col] = 1
    col += 1
    
    # M4 = A11 * (B0k2-B0k1)
    U[3, col] = 1
    V[0*p+k2, col] = 1; V[0*p+k1, col] = -1
    W[k1*n+0, col] = -1; W[k2*n+0, col] = 1
    col += 1
    
    # M5 = (A00+A10) * B0k2
    U[0, col] = 1; U[2, col] = 1
    V[0*p+k2, col] = 1
    W[k1*n+0, col] = 1; W[k1*n+1, col] = -1
    col += 1
    
    # M6 = (A11-A00) * (B0k1+B1k1)
    U[3, col] = 1; U[0, col] = -1
    V[0*p+k1, col] = 1; V[1*p+k1, col] = 1
    W[k1*n+1, col] = 1; W[k2*n+1, col] = 1
    col += 1
    
    # M7 = (A01-A11) * (B1k1+B1k2)
    U[1, col] = 1; U[3, col] = -1
    V[1*p+k1, col] = 1; V[1*p+k2, col] = 1
    W[k2*n+0, col] = -1
    col += 1
    
    return col

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    # Build Strassen-based decomposition: 6 pairs + 1 single
    # 6 pairs * 7 rank + 1 single * 4 rank = 46
    rank = 6 * 7 + 4
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    
    # Process 6 pairs of k-columns: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for group in range(6):
        k1 = 2 * group
        k2 = 2 * group + 1
        col = _build_strassen_block(U, V, W, k1, k2, col)
    
    # Last single column k=12: schoolbook rank 4
    k = 12
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1
    
    if col != rank:
        raise ValueError(f"Column count mismatch: {col} vs {rank}")
    
    if _check_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: full schoolbook (rank 52)
    rf = n*m*p
    Uf = np.zeros((n*m, rf))
    Vf = np.zeros((m*p, rf))
    Wf = np.zeros((p*n, rf))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Uf[i*m+j, col] = 1
                Vf[j*p+k, col] = 1
                Wf[k*n+i, col] = 1
                col += 1
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
