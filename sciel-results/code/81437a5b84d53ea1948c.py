"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, r] = coefficient for A[i,j] in factor r
    V[j*2+k, r] = coefficient for B[j,k] in factor r
    W[k*2+i, r] = coefficient for C[i,k] in factor r
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # P1 = (a11+a22)(b11+b22)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # P2 = (a21+a22)b11
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # P3 = a11(b12-b22)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # P4 = a22(b21-b11)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # P5 = (a11+a12)b22
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # P6 = (a21-a11)(b11+b12)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # P7 = (a12-a22)(b21+b22)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0
    
    return U, V, W

def _schoolbook_3x3x6():
    """Schoolbook decomposition for 3x3x6."""
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _schoolbook_3x2x6():
    """Schoolbook decomposition for 3x2x6."""
    n, m, p = 3, 2, 6
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tolerance=1e-9):
    """Verify the tensor decomposition."""
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False, float('inf')
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False, float('inf')
    
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                error = abs(val - 1.0)
                max_error = max(max_error, error)
    
    return max_error < tolerance, max_error

def _construct_subdivide_conquer():
    """
    Subdivide and conquer: split middle index 5 into 3+2.
    3x5x6 = (3x3x6) + (3x2x6)
    """
    n, m, p = 3, 5, 6
    
    # Part 1: 3x3x6 (middle indices j=0,1,2)
    U1, V1, W1 = _schoolbook_3x3x6()
    rank1 = U1.shape[1]
    
    # Part 2: 3x2x6 (middle indices j=3,4)
    U2, V2, W2 = _schoolbook_3x2x6()
    rank2 = U2.shape[1]
    
    total_rank = rank1 + rank2
    
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    
    # Place part 1 (j=0,1,2)
    for i in range(n):
        for j in range(3):
            U[i*m+j, :rank1] = U1[i*3+j, :]
    for j in range(3):
        for k in range(p):
            V[j*p+k, :rank1] = V1[j*6+k, :]
    for i in range(n):
        for k in range(p):
            W[k*n+i, :rank1] = W1[k*3+i, :]
    
    # Place part 2 (j=3,4)
    for i in range(n):
        for j in range(2):
            U[i*m+(j+3), rank1:] = U2[i*2+j, :]
    for j in range(2):
        for k in range(p):
            V[(j+3)*p+k, rank1:] = V2[j*6+k, :]
    for i in range(n):
        for k in range(p):
            W[k*n+i, rank1:] = W2[k*3+i, :]
    
    return U, V, W

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=5, p=6.
    """
    n, m, p = DIMENSIONS
    
    # Use subdivide and conquer
    U, V, W = _construct_subdivide_conquer()
    
    # Verify
    is_valid, max_error = _verify_decomposition(U, V, W, n, m, p)
    
    if not is_valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
