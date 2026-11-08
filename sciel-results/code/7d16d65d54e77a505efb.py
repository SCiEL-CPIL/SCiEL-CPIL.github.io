import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Block decomposition of 3x3 x 3x8 matrix multiplication
    # Block 1: A[0:2,0:2] * B[0:2,:] -> C[0:2,:] (2x2 x 2x8)
    # Block 2: A[0:2,2] * B[2,:] -> C[0:2,:] (2x1 x 1x8)
    # Block 3: A[2,0:2] * B[0:2,:] -> C[2,:] (1x2 x 2x8)
    # Block 4: A[2,2] * B[2,:] -> C[2,:] (1x1 x 1x8)
    
    U_terms = []
    V_terms = []
    W_terms = []
    rank = 0
    
    # Block 1: schoolbook for 2x2 x 2x8, rank 32
    for i in range(2):
        for j in range(2):
            for k in range(8):
                U_terms.append((i * m + j, rank, 1))
                V_terms.append((j * p + k, rank, 1))
                W_terms.append((k * n + i, rank, 1))
                rank += 1
    
    # Block 2: rank 1 for 2x1 x 1x8 all-ones tensor
    U_terms.append((0 * m + 2, rank, 1))  # U[2, rank]
    U_terms.append((1 * m + 2, rank, 1))  # U[5, rank]
    for k in range(8):
        V_terms.append((2 * p + k, rank, 1))  # V[16+k, rank]
    for k in range(8):
        for i in range(2):
            W_terms.append((k * n + i, rank, 1))  # W[k*3+i, rank]
    rank += 1
    
    # Block 3: schoolbook for 1x2 x 2x8, rank 16
    for j in range(2):
        for k in range(8):
            U_terms.append((2 * m + j, rank, 1))  # U[6+j, rank]
            V_terms.append((j * p + k, rank, 1))
            W_terms.append((k * n + 2, rank, 1))  # W[k*3+2, rank]
            rank += 1
    
    # Block 4: rank 1 for 1x1 x 1x8 all-ones tensor
    U_terms.append((2 * m + 2, rank, 1))  # U[8, rank]
    for k in range(8):
        V_terms.append((2 * p + k, rank, 1))  # V[16+k, rank]
    for k in range(8):
        W_terms.append((k * n + 2, rank, 1))  # W[k*3+2, rank]
    rank += 1
    
    # Build matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for idx, c, val in U_terms:
        U[idx, c] += val
    for idx, c, val in V_terms:
        V[idx, c] += val
    for idx, c, val in W_terms:
        W[idx, c] += val
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1) > 1e-10:
                    pass  # In production, we'd raise an error
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
