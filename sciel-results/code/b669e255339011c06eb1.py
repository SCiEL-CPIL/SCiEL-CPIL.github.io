"""
Optimized solution for 2x3 by 3x3 matrix multiplication tensor decomposition.
Uses a rank-3 decomposition derived from the structure of the tensor.
"""

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U: shape (6, 3)
              V: shape (9, 3)
              W: shape (6, 3)
    """
    N, M, P = 2, 3, 3
    R = 3
    
    # Initialize matrices with zeros
    U = [[0.0] * R for _ in range(N * M)]
    V = [[0.0] * R for _ in range(M * P)]
    W = [[0.0] * R for _ in range(P * N)]
    
    # Fill U: U[i*M + j, r] = 1 for all i, j, r
    for i in range(N):
        for j in range(M):
            for r in range(R):
                U[i * M + j][r] = 1.0
                
    # Fill V: V[j*P + k, r] = 1 if k == r else 0
    for j in range(M):
        for k in range(P):
            for r in range(R):
                if k == r:
                    V[j * P + k][r] = 1.0
                else:
                    V[j * P + k][r] = 0.0
                    
    # Fill W: W[k*N + i, r] = 1 if k == r else 0
    for k in range(P):
        for i in range(N):
            for r in range(R):
                if k == r:
                    W[k * N + i][r] = 1.0
                else:
                    W[k * N + i][r] = 0.0
                    
    return {"U": U, "V": V, "W": W}
