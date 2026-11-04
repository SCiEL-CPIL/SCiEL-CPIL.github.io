import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 20 using Laser Method on Z_5
    rank = 20
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=complex)
    V = np.zeros((m * p, rank), dtype=complex)
    W = np.zeros((p * n, rank), dtype=complex)
    
    # Group Z_5
    G_size = 5
    omega = np.exp(2j * np.pi / G_size)
    
    # Indices for 3x3
    I = [0, 1, 2]
    J = [0, 1, 2]
    K = [0, 1, 2]
    
    col = 0
    # Characters of Z_5: c = 0, 1, 2, 3, 4
    for c in range(G_size):
        # Non-zero elements x in Z_5: r = 1, 2, 3, 4
        for r in range(1, G_size):
            # Construct u, v, w
            # u_{c,r} is supported on (i, j) where j = i + r mod 5
            for i in I:
                j = (i + r) % G_size
                if j in J:
                    U[i * m + j, col] += omega ** (i * c)
            
            # v_{c,r} is supported on (j, k) where k = j - r mod 5
            for j in J:
                k = (j - r) % G_size
                if k in K:
                    V[j * p + k, col] += omega ** (j * c)
            
            # w_{c,r} is supported on (k, i) where i = k + r mod 5
            for k in K:
                i = (k + r) % G_size
                if i in I:
                    W[k * n + i, col] += omega ** (k * c)
            
            col += 1
    
    # Verify the decomposition
    # Reconstruct the tensor T
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    T_reconstructed = np.zeros((n, m, p), dtype=complex)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j, :]
                v_val = V[j * p + k, :]
                w_val = W[k * n + i, :]
                T_reconstructed[i, j, k] = np.sum(u_val * v_val * w_val)
    
    # Expected tensor: T[i, j, k] = 1 for all i, j, k?
    # No, the tensor is 1 if the indices match the multiplication pattern.
    # In the flattened index convention used by the seed:
    # U index: i*m+j
    # V index: j*p+k
    # W index: k*n+i
    # The term A_ij B_jk contributes to C_ki.
    # So T[i, j, k] should be 1 for all i, j, k.
    
    T_expected = np.ones((n, m, p), dtype=complex)
    
    if not np.allclose(T_reconstructed, T_expected, atol=1e-9):
        # Fallback to schoolbook rank 27
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
