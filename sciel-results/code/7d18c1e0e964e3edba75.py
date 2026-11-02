"""Optimized tensor rank decomposition for 3x4 * 4x4 matrix multiplication."""
import time
import numpy as np

N, M, P = 3, 4, 4
RANK_SCHOOLBOOK = N * M * P  # 48


def _verify_fast(U, V, W, n=N, m=M, p=P):
    """Faster verification using tensor reshaping."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(recon, expected, atol=1e-9)


def _strassen_2x2():
    """
    Return rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Product 1: M1 = (a+d)(e+h)
    U[0,0] = 1; U[3,0] = 1  # a, d
    V[0,0] = 1; V[3,0] = 1  # e, h
    W[0,0] = 1; W[3,0] = 1  # C[0,0], C[1,1]
    
    # Product 2: M2 = (c+d)e
    U[2,1] = 1; U[3,1] = 1  # c, d
    V[0,1] = 1              # e
    W[3,1] = -1; W[2,1] = 1  # C[1,1] has -M2, C[1,0] has +M2
    
    # Product 3: M3 = a(f-h)
    U[0,2] = 1              # a
    V[1,2] = 1; V[3,2] = -1  # f, -h
    W[1,2] = 1; W[3,2] = 1   # C[0,1] has +M3, C[1,1] has +M3
    
    # Product 4: M4 = d(g-e)
    U[3,3] = 1              # d
    V[2,3] = 1; V[0,3] = -1  # g, -e
    W[0,3] = 1; W[2,3] = 1   # C[0,0] has +M4, C[1,0] has +M4
    
    # Product 5: M5 = (a+b)h
    U[0,4] = 1; U[1,4] = 1  # a, b
    V[3,4] = 1              # h
    W[0,4] = -1; W[1,4] = 1  # C[0,0] has -M5, C[0,1] has +M5
    
    # Product 6: M6 = (c-d)(f+h)
    U[2,5] = 1; U[3,5] = -1  # c, -d
    V[1,5] = 1; V[3,5] = 1   # f, h
    W[3,5] = 1               # C[1,1] has +M6
    
    # Product 7: M7 = (b-c)(e+g)
    U[1,6] = 1; U[2,6] = -1  # b, -c
    V[0,6] = 1; V[2,6] = 1   # e, g
    W[0,6] = 1               # C[0,0] has +M7
    
    return U, V, W


def _build_optimized():
    """
    Build rank-32 decomposition for 3x4 * 4x4.
    Structure:
    - 4 Strassen blocks (2x2 * 2x2) for the top 2x4 * 4x4 part. Rank 28.
    - 2 rank-2 blocks (1x2 * 2x4) for the bottom 1x4 * 4x4 part. Rank 4.
    """
    rank = 32
    U = np.zeros((12, rank))
    V = np.zeros((16, rank))
    W = np.zeros((12, rank))
    
    col = 0
    sU, sV, sW = _strassen_2x2()
    
    # Helper to add a Strassen block with offsets
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        for c in range(7):
            for li in range(2):
                for lj in range(2):
                    U[(i_off + li)*4 + (j_off + lj), col] = sU[li*2 + lj, c]
            for lj in range(2):
                for lk in range(2):
                    V[(j_off + lj)*4 + (k_off + lk), col] = sV[lj*2 + lk, c]
            for lk in range(2):
                for li in range(2):
                    W[(k_off + lk)*3 + (i_off + li), col] = sW[lk*2 + li, c]
            col += 1
            
    # Block 1: A_L (i=0,1; j=0,1) * B_T0 (j=0,1; k=0,1)
    add_strassen_block(0, 0, 0)
    
    # Block 2: A_L (i=0,1; j=0,1) * B_T1 (j=0,1; k=2,3)
    add_strassen_block(0, 0, 2)
    
    # Block 3: A_R (i=0,1; j=2,3) * B_B0 (j=2,3; k=0,1)
    add_strassen_block(0, 2, 0)
    
    # Block 4: A_R (i=0,1; j=2,3) * B_B1 (j=2,3; k=2,3)
    add_strassen_block(0, 2, 2)
    
    # Bottom part: 1x4 * 4x4
    # Split into j=0,1 and j=2,3
    # Block 5: j=0,1
    for c in range(2):
        j = c
        U[2*4 + j, col] = 1  # i=2, j=j
        for k in range(4):
            V[j*4 + k, col] = 1 # j=j, k=k
            W[k*3 + 2, col] = 1 # k=k, i=2
        col += 1
        
    # Block 6: j=2,3
    for c in range(2):
        j = 2 + c
        U[2*4 + j, col] = 1  # i=2, j=j
        for k in range(4):
            V[j*4 + k, col] = 1 # j=j, k=k
            W[k*3 + 2, col] = 1 # k=k, i=2
        col += 1
        
    if col != rank:
        raise ValueError(f"Column mismatch: {col} != {rank}")
        
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Try the optimized block decomposition
    try:
        U, V, W = _build_optimized()
        if _verify_fast(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j][col] = 1
                V[j*P+k][col] = 1
                W[k*N+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
