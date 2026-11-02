"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = np.zeros((p * n, n * m, m * p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    rank = len(U[0])
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += Ua[u, c] * Va[v, c] * Wa[io, c]
                if s != T[io, u, v]:
                    return False
    return True


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _winograd_fft_4x4x9():
    """
    Winograd FFT-based decomposition for 4x4 * 4x9.
    
    Strategy: Apply 2-point DFT along the 4 output rows.
    Partition rows into pairs (0,2) and (1,3).
    
    For pair (0,2):
      S02[k] = C[0,k] + C[2,k] = sum_j (A[0,j] + A[2,j]) * B[j,k]
      D02[k] = C[0,k] - C[2,k] = sum_j (A[0,j] - A[2,j]) * B[j,k]
    
    For pair (1,3):
      S13[k] = C[1,k] + C[3,k] = sum_j (A[1,j] + A[3,j]) * B[j,k]
      D13[k] = C[1,k] - C[3,k] = sum_j (A[1,j] - A[3,j]) * B[j,k]
    
    Each of S02, D02, S13, D13 is a 1x9 vector computed as dot product of a 4-vector with 4x9 B.
    This requires 4 * 4 * 9 = 144 scalar multiplications at rank 144 (schoolbook for 1x4x9 is 36 each).
    
    But we can do better by also applying FFT along B's columns if p were even.
    Since p=9 is odd, we cannot fully exploit FFT on B.
    
    Alternative: Use the fact that the 4x4x9 tensor can be decomposed via
    Kronecker product structure. 4 = 2*2, so A is a 2x2 block of 2x2 matrices.
    
    Let A = [[A00, A01], [A10, A11]] where each Ak1 is 2x2.
    Let B be partitioned by rows: B = [[B0], [B1]] where B0, B1 are 2x9.
    
    Then C = A*B:
    C[0:2, :] = A00*B0 + A01*B1
    C[2:4, :] = A10*B0 + A11*B1
    
    Each 2x2 * 2x9 multiplication: rank = 2*2*9 = 36 (schoolbook).
    Total: 4 * 36 = 144. Same as before.
    
    The lower bound is max(4*4, 4*9, 9*4) = max(16, 36, 36) = 36.
    So rank 36 is the theoretical minimum.
    
    Let's try to achieve rank 36 directly.
    """
    n, m, p = 4, 4, 9
    # Try rank = 36 = m*p = n*p
    rank = m * p  # 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index: c = j * p + k for j in 0..3, k in 0..8
    # For each (i, j, k): T[k*n+i, i*m+j, j*p+k] = 1
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 iff c corresponds to (j,k)
    # Set: U[i*m+j, j*p+k] = 1, V[j*p+k, j*p+k] = 1, W[k*n+i, j*p+k] = 1
    # But then for fixed (i,j,k): only c=j*p+k contributes, giving 1. Good.
    # For other (i',j',k') with same c=j*p+k: 
    #   U[i'*m+j', j*p+k] must be 0 unless j'=j
    #   W[k'*n+i', j*p+k] must be 0 unless k'=k and i'=i
    # This works! Each (j,k) pair defines one rank-1 term.
    
    c = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
            c += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank 144 is known to be correct; skip the broken rank-36 attempt
    best = _schoolbook(n, m, p)
    return best

# EVOLVE-BLOCK-END
