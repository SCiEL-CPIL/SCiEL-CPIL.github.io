import time
import numpy as np

N = 5
M = 5
P = 5
TIME_LIMIT = 150
SAFETY_MARGIN = 5

def _build_rank125():
    # Standard (schoolbook) rank-125 decomposition for 5x5 x 5x5 -> 5x5.
    # U: (n*m, rank) = (25, 125)
    # V: (m*p, rank) = (25, 125)
    # W: (p*n, rank) = (25, 125)
    # Indexing:
    #   U[i*m + j, t] = 1 if t corresponds to (i, j, k)
    #   V[j*p + k, t] = 1 if t corresponds to (i, j, k)
    #   W[k*n + i, t] = 1 if t corresponds to (i, j, k)
    # where t = i*M*P + j*P + k
    
    U = np.zeros((N*M, N*M*P), dtype=np.float64)
    V = np.zeros((M*P, N*M*P), dtype=np.float64)
    W = np.zeros((P*N, N*M*P), dtype=np.float64)
    
    for i in range(N):
        for j in range(M):
            for k in range(P):
                t = i*M*P + j*P + k
                U[i*M + j, t] = 1.0
                V[j*P + k, t] = 1.0
                W[k*N + i, t] = 1.0
                
    return U, V, W

def construct():
    start_time = time.time()
    
    # Deadline guard
    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    if time_left() < SAFETY_MARGIN:
        # Should not happen for static construction, but safe fallback
        U, V, W = _build_rank125()
        return {'U': U, 'V': V, 'W': W}

    # For 5x5 matrix multiplication, the schoolbook rank is 125.
    # We return the exact schoolbook decomposition.
    
    U, V, W = _build_rank125()
    
    # Verify shapes before returning to avoid ValueError
    # Expected: U (25, 125), V (25, 125), W (25, 125)
    if U.shape[0] != N*M or V.shape[0] != M*P or W.shape[0] != P*N:
        raise ValueError("Internal shape error")

    return {'U': U, 'V': V, 'W': W}
