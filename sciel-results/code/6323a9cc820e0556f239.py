"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time
import itertools
from fractions import Fraction

# EVOLVE-BLOCK-START

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_decomposition(n, m, p, rank, timeout=10):
    """Try to find a decomposition of given rank using random initialization and refinement."""
    start_time = time.time()
    
    # Try multiple random initializations
    for attempt in range(200):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization with small integers
        U = np.zeros((n*m, rank), dtype=object)
        V = np.zeros((m*p, rank), dtype=object)
        W = np.zeros((p*n, rank), dtype=object)
        
        # Initialize with random ±1 or 0
        for i in range(n*m):
            for r in range(rank):
                if np.random.random() < 0.3:
                    U[i, r] = np.random.choice([-1, 1])
        for i in range(m*p):
            for r in range(rank):
                if np.random.random() < 0.3:
                    V[i, r] = np.random.choice([-1, 1])
        for i in range(p*n):
            for r in range(rank):
                if np.random.random() < 0.3:
                    W[i, r] = np.random.choice([-1, 1])
        
        # Convert to lists for the interface
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _validate(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _strassen_2x2(A, B):
    """Strassen's algorithm for 2x2 matrix multiplication."""
    # A and B are 2x2
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]
    e, f = B[0][0], B[0][1]
    g, h = B[1][0], B[1][1]
    
    m1 = (a + d) * (e + h)
    m2 = (c + d) * e
    m3 = a * (f - h)
    m4 = d * (g - e)
    m5 = (a + b) * h
    m6 = (c - a) * (e + f)
    m7 = (b - d) * (g + h)
    
    C = [[m1 + m4 - m5 + m7, m3 + m5],
         [m2 + m4, m1 - m2 + m3 + m6]]
    return C

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # Known results suggest we might be able to do better than 48
    
    # Try ranks from 24 up to 47
    for rank in range(24, min(48, best_rank)):
        if time.time() > deadline - 30:  # Safety margin
            break
        
        result = _try_decomposition(n, m, p, rank, timeout=5)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result

# EVOLVE-BLOCK-END
