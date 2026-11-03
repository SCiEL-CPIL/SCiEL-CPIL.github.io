import time

DIMENSIONS = [2, 4, 6]
N, M, P = 2, 4, 6
NM = N * M  # 8
MP = M * P  # 24
PN = P * N  # 12

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Algebraic construction: rank = M = 4
    # U[i*M+j, c] = 1 if c == j, else 0
    # V[j*P+k, c] = 1 if c == j, else 0  
    # W[k*N+i, c] = 1 if c == j... wait, W depends on i and k, not j directly.
    
    # Let me re-derive. We need:
    # sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c] = 1 for all i,j,k
    #
    # Set V[j*P+k, c] = 1 if c == j, else 0. (V has 1s on "diagonal" indexed by j)
    # Then sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c] 
    #   = U[i*M+j, j] * V[j*P+k, j] * W[k*N+i, j]
    #   = U[i*M+j, j] * 1 * W[k*N+i, j]
    # We need U[i*M+j, j] * W[k*N+i, j] = 1 for all i, j, k.
    # So set U[i*M+j, j] = 1 and W[k*N+i, j] = 1 for all valid i,j,k.
    
    rank = M  # 4
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    for i in range(N):
        for j in range(M):
            U[i * M + j][j] = 1
            for k in range(P):
                V[j * P + k][j] = 1
                W[k * N + i][j] = 1
    
    # Verify the decomposition
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = sum(U[i*M+j][c] * V[j*P+k][c] * W[k*N+i][c] for c in range(rank))
                if val != 1:
                    # Fallback to schoolbook
                    rank = N * M * P
                    U = [[0] * rank for _ in range(NM)]
                    V = [[0] * rank for _ in range(MP)]
                    W = [[0] * rank for _ in range(PN)]
                    col = 0
                    for ii in range(N):
                        for jj in range(M):
                            for kk in range(P):
                                U[ii * M + jj][col] = 1
                                V[jj * P + kk][col] = 1
                                W[kk * N + ii][col] = 1
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
