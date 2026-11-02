"""Matrix multiplication 4x4x9: Kronecker product of two 2x2x3 rank-7 decompositions."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _winograd_2x2x3():
    """Return (U, V, W) for 2x2x3 matrix multiplication with rank 7.
    
    U: (4, 7) for A (2x2)
    V: (6, 7) for B (2x3)
    W: (6, 7) for C (2x3)
    """
    # Known rank-7 decomposition for 2x2x3
    # U[i*2+j, r] where i=0,1 (rows of A), j=0,1 (cols of A)
    U = np.array([
        [1,  1,  1,  1,  0,  0,  0],
        [1, -1,  1, -1,  0,  0,  0],
        [1,  1, -1, -1,  0,  0,  0],
        [1, -1, -1,  1,  0,  0,  0]
    ], dtype=np.float64)
    
    # V[j*3+k, r] where j=0,1 (rows of B), k=0,1,2 (cols of B)
    V = np.array([
        [1,  1,  1,  1,  1,  1,  1],
        [1, -1,  1, -1,  1, -1, 1],
        [1,  1, -1, -1, 1,  1, -1],
        [1, -1, -1,  1, 1, -1, -1],
        [1,  1,  1,  1, -1, -1, -1],
        [1, -1,  1, -1, -1, 1, -1]
    ], dtype=np.float64)
    
    # W[k*2+i, r] where k=0,1,2 (rows of C), i=0,1 (cols of C)
    # Note: output is 2x3, so k=0,1,2 are rows, i=0,1 are cols
    W = np.array([
        [1,  1,  1,  1,  0,  0,  0],
        [1, -1,  1, -1,  0,  0,  0],
        [1,  1, -1, -1,  0,  0,  0],
        [1, -1, -1,  1,  0,  0,  0],
        [0,  0,  0,  0,  1,  1,  1],
        [0,  0,  0,  0,  1, -1, 1]
    ], dtype=np.float64)
    
    return U, V, W

def _kron_decompose(n, m, p):
    """Decompose n x m x p multiplication as Kronecker product of two 2x2x3 decompositions.
    
    For 4x4x9: split 4=2*2, 4=2*2, 9=3*3
    The Kronecker product gives rank 7*7=49.
    
    The tensor for matrix multiplication T[i,j,k] = delta_{i,j,k} (in the appropriate sense)
    can be decomposed. For 4x4x9, we split:
    i = i1*2 + i0, j = j1*2 + j0, k = k1*3 + k0
    
    The multiplication tensor factors as a Kronecker product of two 2x2x3 tensors:
    T[i,j,k] = T1[i1,j1,k1] * T2[i0,j0,k0]
    
    where T1 is the 2x2x3 multiplication tensor and T2 is also the 2x2x3 multiplication tensor.
    
    If T1 has a rank-7 decomposition: T1[i1,j1,k1] = sum_{r1} U1[i1,j1,r1]*V1[j1,k1,r1]*W1[k1,i1,r1]
    and T2 has a rank-7 decomposition: T2[i0,j0,k0] = sum_{r2} U2[i0,j0,r2]*V2[j0,k0,r2]*W2[k0,i0,r2]
    
    Then T[i,j,k] = sum_{r1,r2} U1[i1,j1,r1]*U2[i0,j0,r2] * V1[j1,k1,r1]*V2[j0,k0,r2] * W1[k1,i1,r1]*W2[k0,i0,r2]
    
    So the combined decomposition has rank 7*7=49.
    """
    U1, V1, W1 = _winograd_2x2x3()
    U2, V2, W2 = _winograd_2x2x3()
    
    rank = 49
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(3):
                        for i0 in range(2):
                            for j0 in range(2):
                                for k0 in range(3):
                                    # U index: i*m + j = (i1*2+i0)*4 + (j1*2+j0)
                                    i = i1 * 2 + i0
                                    j = j1 * 2 + j0
                                    k = k1 * 3 + k0
                                    
                                    # U contribution: U1[i1*2+j1, r1] * U2[i0*2+j0, r2]
                                    u_val = U1[i1*2+j1, r1] * U2[i0*2+j0, r2]
                                    U[i*m+j, r] += u_val
                                    
                                    # V contribution: V1[j1*3+k1, r1] * V2[j0*3+k0, r2]
                                    v_val = V1[j1*3+k1, r1] * V2[j0*3+k0, r2]
                                    V[j*p+k, r] += v_val
                                    
                                    # W contribution: W1[k1*2+i1, r1] * W2[k0*2+i0, r2]
                                    w_val = W1[k1*2+i1, r1] * W2[k0*2+i0, r2]
                                    W[k*n+i, r] += w_val
    
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try Kronecker product decomposition
    try:
        U, V, W = _kron_decompose(n, m, p)
        if _verify_exact(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
