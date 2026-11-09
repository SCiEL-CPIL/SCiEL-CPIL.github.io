import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    n, m, p = 3, 3, 5
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    def verify_decomposition(U, V, W, rank):
        # Check if U @ V^T * W correctly decomposes the tensor
        # For matrix multiplication: (A * B)[i, k] = sum_j A[i, j] * B[j, k]
        # Tensor form: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need to verify this equals 1 if j is "connected" else 0
        # Actually, the tensor for matrix multiplication M_{i,j,k} = delta_{connected}
        # For 3x3 * 3x5: A is 3x3, B is 3x5, C = A*B is 3x5
        # C[i, k] = sum_j A[i, j] * B[j, k]
        # So the tensor T[i, j, k] should be 1 for all valid (i, j, k)
        # Wait, no. The tensor decomposition represents the bilinear map.
        # C[i, k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i, j] * B[j, k]
        # This must equal sum_j A[i, j] * B[j, k]
        # So for each (i, k), sum_j (sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i, j] * B[j, k]) = sum_j A[i, j] * B[j, k]
        
        # Let's construct the full tensor and compare
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This T[i, j, k] must equal 1 for all i, j, k (since A[i,j]*B[j,k] are independent variables)
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    # The tensor entry should be 1
                    error += (val - 1.0) ** 2
        return error
    
    # Start with the trivial rank-45 decomposition
    rank = n * m * p
    U = [[0.0] * rank for _ in range(size_U)]
    V = [[0.0] * rank for _ in range(size_V)]
    W = [[0.0] * rank for _ in range(size_W)]
    
    col_idx = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U[i*m + j][col_idx] = 1.0
                V[j*p + k][col_idx] = 1.0
                W[k*n + i][col_idx] = 1.0
                col_idx += 1
    
    # Try to reduce rank by finding zero rows/columns in the trivial decomposition
    # and removing them, then re-optimizing
    
    # For now, return the trivial decomposition
    return {"U": U, "V": V, "W": W}
