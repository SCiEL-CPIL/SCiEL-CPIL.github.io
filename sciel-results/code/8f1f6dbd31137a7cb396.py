"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
    
    def verify(U, V, W, tol=1e-6):
        # Check if U, V, W form a valid tensor decomposition
        # T[i, j, k, l] = sum_c U[i, c] * V[j, c] * W[k, l]
        # The tensor for matrix multiplication (A * B) is:
        # T[i, j, k, l] = 1 if j == l else 0  (where i is row of A, j is col of A/row of B, k is col of B, l is row of result)
        # Wait, the convention is:
        # U has shape (n*m, rank) where row index is i*m + j
        # V has shape (m*p, rank) where row index is j*p + k
        # W has shape (p*n, rank) where row index is k*n + l
        # The tensor is T[i, j, k, l] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+l, c]
        # And we want T[i, j, k, l] = delta_{j,l}
        
        # Let's check a few random indices
        # Or check all? 4*4*9*4 = 576 terms. That's fast.
        # Actually, let's check if the tensor is correct.
        # The tensor should be the identity in the (j, l) indices, independent of i, k.
        # So T[i, j, k, l] = 1 if j == l else 0.
        
        # We can form the tensor and check.
        # But forming the full tensor is 4*4*9*4 = 576 numbers, which is small.
        # However, the rank can be large (e.g. 144).
        # Let's just check a few random (i, j, k, l) tuples.
        for _ in range(1000):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            l = random.randint(0, n - 1)
            # Compute sum
            s = 0.0
            for c in range(min(rank_school, U.shape[1])):
                u_val = U[i * m + j, c]
                if u_val != 0:
                    v_val = V[j * p + k, c]
                    if v_val != 0:
                        w_val = W[k * n + l, c]
                        if w_val != 0:
                            s += u_val * v_val * w_val
            expected = 1.0 if j == l else 0.0
            if abs(s - expected) > tol:
                return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Schoolbook decomposition is invalid")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
