import numpy as np
import time
import sys
from functools import lru_cache

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """Return the Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns (U, V, W) where:
    - U has shape (4, 7) for the 2x2 input matrix A
    - V has shape (4, 7) for the 2x2 input matrix B  
    - W has shape (4, 7) for the 2x2 output matrix C
    
    The convention is: C[i*n+j] = sum_k A[i*m+k] * B[k*n+j]
    For 2x2: A is indexed 0..3 (row-major), B is 0..3, C is 0..3
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # Products:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    #
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # U matrix: rows are A entries (a00, a01, a10, a11), columns are the 7 products
    # V matrix: rows are B entries (b00, b01, b10, b11), columns are the 7 products
    # W matrix: rows are C entries (c00, c01, c10, c11), columns are the 7 products
    
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0],  # a00
        [0, 0, 0, 0, 1, -1, 1], # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [1, 0, 1, 0, 0, 1, 0],   # b01
        [1, 0, -1, 1, 1, 0, 1],  # b10
        [1, 0, 0, 0, 1, 0, 1]    # b11
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ], dtype=float)
    
    return U, V, W

def _decompose(n, m, p):
    """Recursively decompose the n x m by m x p matrix multiplication tensor.
    
    Returns (U, V, W) where:
    - U has shape (n*m, rank)
    - V has shape (m*p, rank) 
    - W has shape (p*n, rank)
    
    Uses Strassen for 2x2x2 blocks, schoolbook otherwise.
    """
    # Base case: schoolbook
    if n <= 1 and m <= 1 and p <= 1:
        rank = 1
        U = np.zeros((1, 1))
        V = np.zeros((1, 1))
        W = np.zeros((1, 1))
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        return U, V, W
    
    # If all dimensions are 2, use Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Try to split along the middle dimension (m) if it's even
    if m % 2 == 0:
        m1 = m // 2
        m2 = m - m1
        # Split: A is n x m, B is m x p
        # A = [A1; A2] where A1 is n x m1, A2 is n x m2
        # B = [B1, B2] where B1 is m1 x p, B2 is m2 x p
        # C = A1*B1 + A2*B2
        
        U1, V1, W1 = _decompose(n, m1, p)
        U2, V2, W2 = _decompose(n, m2, p)
        
        # Combine
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # First part: A1 * B1
        # A1 entries are at indices [i*m + j] for j in [0, m1)
        # B1 entries are at indices [j*p + k] for j in [0, m1)
        # W is the same
        
        for i in range(n):
            for j in range(m1):
                U[i * m + j, :U1.shape[1]] = U1[i * m1 + j, :]
        
        for j in range(m1):
            for k in range(p):
                V[j * p + k, :V1.shape[1]] = V1[j * p + k, :]
        
        W[:, :W1.shape[1]] = W1
        
        # Second part: A2 * B2
        # A2 entries are at indices [i*m + j] for j in [m1, m)
        # B2 entries are at indices [j*p + k] for j in [m1, m)
        
        for i in range(n):
            for j in range(m1, m):
                U[i * m + j, U1.shape[1]:] = U2[i * m2 + (j - m1), :]
        
        for j in range(m1, m):
            for k in range(p):
                V[j * p + k, V1.shape[1]:] = V2[(j - m1) * p + k, :]
        
        W[:, W1.shape[1]:] = W2
        
        return U, V, W
    
    # Try to split along n if it's even
    if n % 2 == 0:
        n1 = n // 2
        n2 = n - n1
        # Split: A is n x m, B is m x p
        # A = [A1; A2] where A1 is n1 x m, A2 is n2 x m
        # C = [C1; C2] where C1 = A1*B, C2 = A2*B
        
        U1, V1, W1 = _decompose(n1, m, p)
        U2, V2, W2 = _decompose(n2, m, p)
        
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # First part: A1 * B
        for i in range(n1):
            for j in range(m):
                U[i * m + j, :U1.shape[1]] = U1[i * m + j, :]
        
        V[:, :V1.shape[1]] = V1
        
        for k in range(p):
            for i in range(n1):
                W[k * n + i, :W1.shape[1]] = W1[k * n1 + i, :]
        
        # Second part: A2 * B
        for i in range(n1, n):
            for j in range(m):
                U[i * m + j, U1.shape[1]:] = U2[(i - n1) * m + j, :]
        
        V[:, V1.shape[1]:] = V2
        
        for k in range(p):
            for i in range(n1, n):
                W[k * n + i, W1.shape[1]:] = W2[k * n2 + (i - n1), :]
        
        return U, V, W
    
    # Try to split along p if it's even
    if p % 2 == 0:
        p1 = p // 2
        p2 = p - p1
        # Split: A is n x m, B is m x p
        # B = [B1, B2] where B1 is m x p1, B2 is m x p2
        # C = [C1, C2] where C1 = A*B1, C2 = A*B2
        
        U1, V1, W1 = _decompose(n, m, p1)
        U2, V2, W2 = _decompose(n, m, p2)
        
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # First part: A * B1
        U[:, :U1.shape[1]] = U1
        
        for j in range(m):
            for k in range(p1):
                V[j * p + k, :V1.shape[1]] = V1[j * p1 + k, :]
        
        for k in range(p1):
            for i in range(n):
                W[k * n + i, :W1.shape[1]] = W1[k * n + i, :]
        
        # Second part: A * B2
        U[:, U1.shape[1]:] = U2
        
        for j in range(m):
            for k in range(p1, p):
                V[j * p + k, V1.shape[1]:] = V2[j * p2 + (k - p1), :]
        
        for k in range(p1, p):
            for i in range(n):
                W[k * n + i, W1.shape[1]:] = W2[(k - p1) * n + i, :]
        
        return U, V, W
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    try:
        U, V, W = _decompose(n, m, p)
    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Verify the decomposition is correct
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition correctly computes matrix multiplication."""
        # For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # And for other combinations, the sum should be 0
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check that other entries are zero (sample some)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i, j, k) != (i2, j2, k2):
                                    # This is the wrong combination
                                    val = np.sum(U[i * m + j, :] * V[j2 * p + k, :] * W[k * n + i2, :])
                                    if abs(val) > 1e-9:
                                        return False
        return True
    
    # Verify
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
