"""Rewritten program for matrix multiplication tensor decomposition M_{3,5,7}."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 3, 5, 7

    def verify_fast(U, V, W):
        """Fast verification using tensor contraction."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        r = U.shape[1]
        T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                k = b % p
                for c in range(p * n):
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    if abs(T[a, b, c] - expected) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, r

    # Start with schoolbook
    U_sb, V_sb, W_sb, base_rank = schoolbook()
    current_best_rank = base_rank
    current_best = (U_sb, V_sb, W_sb)

    # Try structured search to find lower rank
    # Strategy: use random perturbations and gradient descent
    # But for now, we'll try a few structured approaches

    # Approach 1: Try to use the fact that 3 = 1+2, 5 = 2+3, 7 = 3+4
    # This might allow some sharing of multiplications

    # Approach 2: Use Winograd-like blocking
    # For small dimensions, we can try to find low-rank decompositions
    # by exploiting the structure of the tensor

    # For now, let's try a random search with verification
    # This is expensive, so we'll limit the number of trials

    np.random.seed(42)
    
    # Try random low-rank decompositions
    for r in range(50, 106):  # Try ranks from 50 to 105
        if time.time() > deadline - 30:
            break
            
        # Try a few random decompositions for each rank
        for trial in range(10):
            if time.time() > deadline - 30:
                break
                
            # Initialize with random values
            U_rand = np.random.randn(n * m, r)
            V_rand = np.random.randn(m * p, r)
            W_rand = np.random.randn(p * n, r)
            
            # Verify
            if verify_fast(U_rand, V_rand, W_rand):
                current_best_rank = r
                current_best = (U_rand, V_rand, W_rand)
                break

    # Return the best decomposition found
    U, V, W = current_best
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
