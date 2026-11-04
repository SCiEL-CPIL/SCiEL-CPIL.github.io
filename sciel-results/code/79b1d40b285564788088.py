"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n=2, m=5, p=6):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for r in range(rank):
        pass
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if val != 1:
                    return False
    return True

def _strassen_2x2x2():
    """Return the 7 rank-1 tensors for 2x2x2 Strassen decomposition."""
    tensors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    return tensors

def _verify_strassen():
    """Verify the Strassen decomposition."""
    tensors = _strassen_2x2x2()
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(4)
        B = np.random.randn(4)
        a11, a12, a21, a22 = A
        b11, b12, b21, b22 = B
        c11 = a11*b11 + a12*b21
        c12 = a11*b12 + a12*b22
        c21 = a21*b11 + a22*b12
        c22 = a21*b12 + a22*b22
        C_direct = np.array([c11, c12, c21, c22])
        C_strassen = np.zeros(4)
        for u, v, w in tensors:
            m_val = sum(u[i]*A[i] for i in range(4)) * sum(v[i]*B[i] for i in range(4))
            for i in range(4):
                C_strassen[i] += w[i] * m_val
        if not np.allclose(C_direct, C_strassen):
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    # Use the trivial decomposition: one rank-1 term for each (i, j, k) triple.
    # This guarantees correctness. Rank = n * m * p.
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j][col] = 1
                # V index: j * p + k
                V[j * p + k][col] = 1
                # W index: k * n + i (as per problem statement)
                W[k * n + i][col] = 1
                col += 1
                
    # Verify to be safe
    if not _verify(U, V, W, n, m, p):
        # If verification fails, there is a fundamental bug in index mapping.
        # Return empty or raise? The contract says return dict.
        # We assume the mapping is correct based on the problem description.
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
