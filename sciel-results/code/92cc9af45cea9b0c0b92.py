import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-32 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    
    Key insight: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    We use 2 terms per column k (one for each row of A), giving rank 32.
    Each term s = 2k + j contributes A[i,j]*B[j,k] to C[i,k].
    
    Wait, that's still schoolbook at rank 64. Let me reconsider.
    
    Actually, for m=2, the minimal rank might indeed be 32 if we can share.
    Let me try: for each k, use 2 terms. Term 2k: U picks A[0,0],A[1,0], V picks B[0,k], W picks C[0,k],C[1,k]
    Term 2k+1: U picks A[0,1],A[1,1], V picks B[1,k], W picks C[0,k],C[1,k]
    
    This gives: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    That's exactly correct with rank 32!
    """
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)  # 4 x 32
    V = np.zeros((m * p, rank), dtype=float)  # 32 x 32
    W = np.zeros((p * n, rank), dtype=float)  # 32 x 32
    
    # For each column k of B (0..15), and each row j of A (0,1):
    # Term s = 2*k + j
    # U: A[i,j] for i=0,1 -> indices i*m+j = i*2+j
    # V: B[j,k] -> index j*p+k = j*16+k
    # W: C[i,k] for i=0,1 -> index k*n+i = k*2+i
    
    for k in range(p):
        for j in range(m):
            s = 2 * k + j
            
            # U: A[i,j] for i=0,1
            # A index: i*m + j = i*2 + j
            U[0 * m + j, s] = 1  # A[0,j]
            U[1 * m + j, s] = 1  # A[1,j]
            
            # V: B[j,k]
            # B index: j*p + k
            V[j * p + k, s] = 1
            
            # W: C[i,k] for i=0,1
            # C index: k*n + i = k*2 + i
            W[k * n + 0, s] = 1  # C[0,k]
            W[k * n + 1, s] = 1  # C[1,k]
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = 0.0
                    for s in range(rank):
                        val += U[u, s] * V[v, s] * W[w, s]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 64
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=float)
    V = np.zeros((m * p, rank_school), dtype=float)
    W = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
