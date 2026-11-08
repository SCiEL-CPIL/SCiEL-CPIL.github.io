import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    C = A * B, where A is 3x3, B is 3x8, C is 3x8.
    
    Returns:
        A dictionary with keys 'U', 'V', 'W' containing numpy arrays.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Standard bilinear form for matrix multiplication
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We decompose the tensor T_{i,j,k} such that T = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}?
    # Wait, the trace convention is W[k*n+i, r].
    # The standard tensor for C = A*B is T_{i,j,k} = 1 if C_{i,k} depends on A_{i,j} and B_{j,k}.
    # Decomposition: T_{i,j,k} = sum_r u_{i,j}^{(r)} v_{j,k}^{(r)} w_{k,i}^{(r)}
    # Actually, the standard form is:
    # C_{i,k} = sum_j sum_r A_{i,j} B_{j,k} (u_{i,j}^{(r)} v_{j,k}^{(r)} w_{k,i}^{(r)})?
    # No, the rank-1 tensor is u \otimes v \otimes w.
    # The product is sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] * (something)?
    
    # Let's stick to the explicit construction in the seed which is correct.
    # Seed logic:
    # U[i*m+j][c] = 1
    # V[j*p+k][c] = 1
    # W[k*n+i][c] = 1
    # This corresponds to the term A_{i,j} * B_{j,k} * C_{k,i} (where C is the output index, but mapped to W's second index?).
    # Let's verify the convention.
    # Output factor W[k*n+i, r].
    # The multiplication is C = A B.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The tensor is T_{i, j, k}.
    # Decomposition: T_{i, j, k} = sum_r U_{i, j}^{(r)} V_{j, k}^{(r)} W_{k, i}^{(r)}
    # Here U has shape (n*m, rank), indexed by i*m+j.
    # V has shape (m*p, rank), indexed by j*p+k.
    # W has shape (p*n, rank), indexed by k*n+i.
    
    # This matches the seed exactly.
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
