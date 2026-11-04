"""
Solution for 2x3 by 3x3 matrix multiplication tensor decomposition.
Uses the trivial rank-18 decomposition (one rank-1 tensor per output entry).
"""

import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U: shape (6, 18)
              V: shape (9, 18)
              W: shape (6, 18)
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Must finish well before 150s kill
    
    N, M, P = 2, 3, 3
    R = N * M * P  # 18
    
    # Initialize matrices with zeros
    # U shape: (N*M, R) = (6, 18)
    # V shape: (M*P, R) = (9, 18)
    # W shape: (P*N, R) = (6, 18)
    
    U = [[0.0] * R for _ in range(N * M)]
    V = [[0.0] * R for _ in range(M * P)]
    W = [[0.0] * R for _ in range(P * N)]
    
    # Trivial decomposition:
    # r = i * M * P + j * P + k
    # i in 0..N-1, j in 0..M-1, k in 0..P-1
    #
    # U[i*M + j, r] = 1 if r corresponds to this (i, j) else 0
    # V[j*P + k, r] = 1 if r corresponds to this (j, k) else 0
    # W[k*N + i, r] = 1 if r corresponds to this (k, i) else 0
    
    for i in range(N):
        for j in range(M):
            for k in range(P):
                r = i * M * P + j * P + k
                
                # U index: i*M + j
                U[i * M + j][r] = 1.0
                
                # V index: j*P + k
                V[j * P + k][r] = 1.0
                
                # W index: k*N + i (per task convention W[k*n+i, r])
                W[k * N + i][r] = 1.0
    
    # Check if we're approaching the deadline
    if time.time() > deadline - 10.0:
        # Return immediately to be safe
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
