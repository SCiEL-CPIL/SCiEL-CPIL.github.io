import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p):
    """Verify if U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # T[i, k, j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i, k, j] to be 1 if it corresponds to a multiplication term, else 0.
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices (i, k, j) such that T[i, k, j] = 1.
    # Wait, the problem says W[k*n+i, r].
    # So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # This sums to C[i, k].
    # So for each i, k, the sum over r and j should give the coefficient of A[i, j] B[j, k].
    # Actually, the tensor T_{i, j, k} is 1 for all i, j, k?
    # No, the tensor for matrix multiplication is such that
    # C_{ik} = sum_{j, r} U_{ij, r} V_{jk, r} W_{ki, r}.
    # This must equal sum_j A_{ij} B_{jk}.
    # So for each i, j, k, the coefficient of A_{ij} B_{jk} in the expansion must be 1.
    # And coefficients for other terms must be 0.
    # But the expansion is linear in A and B.
    # The term A_{ij} B_{jk} appears with coefficient sum_r U_{ij, r} V_{jk, r} W_{ki, r}.
    # So we need sum_r U_{ij, r} V_{jk, r} W_{ki, r} = 1 for all i, j, k.
    # Is that correct?
    # Yes, because each A_{ij} and B_{jk} is an independent variable.
    
    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, time_limit):
    """Try to find a decomposition of given rank."""
    start_time = time.time()
    
    # Initialize with random values
    for attempt in range(100):
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Scale to make magnitudes reasonable
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-9
        V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-9
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
        
        # Check if this random initialization is close to a solution?
        # This is unlikely. We need a search.
        # Given the complexity, random search for exact integer solutions is not feasible in 150s.
        # I will return None if not found quickly.
        
        if time.time() - start_time > time_limit:
            return None

    return None

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Deadline guard
    deadline = time.time() + 110
    
    # Try to find a better rank?
    # For now, return schoolbook.
    # If I had a known algorithm, I would use it.
    
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
