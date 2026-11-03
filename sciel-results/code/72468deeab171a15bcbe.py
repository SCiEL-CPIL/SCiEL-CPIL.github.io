"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Vectorized verification using einsum
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Reshape to (n, m, R), (m, p, R), (p, n, R)
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    
    # Sum over R: T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    # This is tricky for einsum directly.
    # Let's use a loop over R but vectorized inner product?
    # R is small.
    T = np.zeros((n, m, p), dtype=float)
    R = U.shape[1]
    for r in range(R):
        u = Ur[:, :, r] # (n, m)
        v = Vr[:, :, r] # (m, p)
        w = Wr[:, :, r] # (p, n)
        
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # This is equivalent to:
        # For each i, k: sum_j u[i,j] * v[j,k] * w[k,i]
        # Let M = u @ v (n x p)
        # Then T[i,k] += M[i,k] * w[k,i] ? No, w[k,i] is (p,n).
        # M[i,k] = sum_j u[i,j] v[j,k]
        # Term = M[i,k] * w[k,i]
        # So T[i,k] += M[i,k] * w.T[i,k] ? w[k,i] is element (k,i) of w.
        # w is (p, n). w[k,i] is w[k][i].
        # So we need element-wise product of M (n,p) and w.T (n,p)?
        # w.T has shape (n, p). w.T[i, k] = w[k, i].
        # Yes!
        M = u @ v # (n, p)
        T += M * w.T # (n, p) element-wise
        
    T_target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, T_target, atol=tol)

def _try_strassen_2x2_2x2():
    """
    Attempt to use Strassen's algorithm for 2x2*2x2 as a building block.
    However, for 3x4*4x5, we can't directly use 2x2 blocks without padding.
    Padding to 4x4*4x4 would use rank 49 (Laderman) for the 4x4 case.
    But we only need 3x4*4x5.
    
    Actually, let's try a different approach:
    We can view the 3x4*4x5 multiplication as a set of independent multiplications.
    
    A more promising approach:
    Use the fact that R(3,4,5) might be reducible.
    
    Known results:
    - R(2,2,2) = 7 (Strassen)
    - R(3,3,3) = 23 (Strassen)
    - R(4,4,4) = 49 (Laderman)
    
    For 3x4*4x5, one strategy is to decompose into smaller blocks.
    For example, split the 4x5 matrix into [2x5; 2x5] and the 3x4 into [3x2 | 3x2].
    Then C = A1*B1 + A2*B2 where A1 is 3x2, B1 is 2x5, A2 is 3x2, B2 is 2x5.
    This doesn't directly help because we still need to multiply 3x2 by 2x5.
    
    Another approach: Use the Strassen algorithm for 2x2 blocks.
    Split the 4x4 part into 2x2 blocks. But we have 3x4*4x5.
    
    Let's try to construct a decomposition using Strassen's 7-multiplication trick for 2x2 blocks.
    We can pad the matrices to 4x4*4x4, use Laderman's 49-multiplication algorithm,
    and then extract the relevant 3x4*4x5 part.
    
    Laderman's algorithm for 4x4*4x4 has rank 49.
    If we pad 3x4*4x5 to 4x4*4x4, we can use the 49 rank decomposition.
    But we need to verify that the extracted part is correct.
    
    Actually, this is a valid approach!
    Let's implement Laderman's 4x4*4x4 algorithm and extract the 3x4*4x5 part.
    """
    # Laderman's algorithm for 4x4*4x4
    # This is a known result with rank 49.
    # We'll construct the U, V, W matrices for 4x4*4x4 and then restrict to 3x4*4x5.
    
    n4, m4, p4 = 4, 4, 4
    rank4 = 49
    
    # Laderman's algorithm coefficients
    # This is complex to implement from scratch. Let's use a known construction.
    # For now, let's just return the schoolbook result since implementing Laderman
    # correctly is non-trivial and error-prone.
    
    # Actually, let's try a simpler approach:
    # Use the fact that R(3,4,5) <= R(4,4,4) = 49 by padding.
    # If we can implement a rank-49 decomposition for 4x4*4x4,
    # we can extract the 3x4*4x5 part.
    
    # For now, let's just return None to indicate this approach is not implemented.
    return None

def _try_winograd():
    """
    Try to use Winograd's algorithm for matrix multiplication.
    Winograd's F(2) transform can reduce the number of multiplications.
    For 3x4*4x5, we can try to apply Winograd's method.
    """
    # Winograd's algorithm is complex to implement correctly.
    # For now, let's return None.
    return None

def _random_search(max_rank=59, num_trials=10000):
    """
    Random search for a lower rank decomposition.
    Disabled or limited to prevent timeout.
    """
    # The search space is too large for random guessing to find a valid
    # integer/half-integer decomposition of rank < 60 in reasonable time.
    # We skip this to ensure the program returns within the time limit.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    # Try to find a lower rank decomposition
    # First, try structured approaches
    result = _try_strassen_2x2_2x2()
    if result is not None:
        return result
    
    result = _try_winograd()
    if result is not None:
        return result
    
    # Check time
    if time.time() > deadline - 10:
        return _schoolbook()
    
    # Try random search (currently disabled/short)
    result = _random_search(max_rank=59, num_trials=100)
    if result is not None:
        return result
    
    # Return schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
