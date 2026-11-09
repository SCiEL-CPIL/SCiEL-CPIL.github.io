"""Program for matrix multiplication tensor decomposition (4x5 * 5x5)."""
import time
import random

DIMENSIONS = [4, 5, 5]


def _verify(U, V, W, n, m, p):
    """Check whether U, V, W exactly decompose the matmul tensor."""
    import numpy as np
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Build tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But actually the decomposition is: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] = delta (the matmul tensor) has T[i,j,k] = 1 for all i,j,k
    # and the decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # Let's check this
    # Reshape for efficient computation
    U_r = U.reshape(n, m, rank)  # U_r[i, j, r]
    V_r = V.reshape(m, p, rank)  # V_r[j, k, r]
    W_r = W.reshape(p, n, rank)  # W_r[k, i, r]
    
    # For each (i,j,k): sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # This is a trilinear form. We can compute it as:
    # For fixed i,j: sum_k (sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r])
    # = sum_k (U_r[i,j,:] * V_r[j,k,:]) dot W_r[k,i,:]
    # Let's just do it naively for small dimensions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _try_random_rank(n, m, p, rank, trials=200, timeout=10):
    """Try to find a rank-rank decomposition via random search."""
    import numpy as np
    start = time.time()
    for t in range(trials):
        if time.time() - start > timeout:
            break
        U = np.random.randint(-2, 3, (n*m, rank))
        V = np.random.randint(-2, 3, (m*p, rank))
        W = np.random.randint(-2, 3, (p*n, rank))
        # Check
        ok = True
        for i in range(n):
            if not ok:
                break
            for j in range(m):
                if not ok:
                    break
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if val != 1:
                        ok = False
                        break
        if ok:
            return U.tolist(), V.tolist(), W.tolist()
    return None


def _try_strassen_blocks(n, m, p):
    """
    Try to build a decomposition using Strassen-like block recursion.
    For 4x5 * 5x5, we can try to exploit structure.
    """
    import numpy as np
    # One approach: treat the 4x5 matrix as blocks and use Strassen on sub-blocks
    # Actually, let's try a direct approach: use known lower bounds.
    # The rank of matmul(4,5,5) is at most 4*5*5 = 100 (schoolbook).
    # We can try to do better by finding a decomposition.
    
    # Let's try a simple approach: use the fact that we can write
    # C[i,k] = sum_j A[i,j] B[j,k]
    # as a sum of rank-1 terms with clever linear combinations.
    
    # For now, just return None and let the main function use schoolbook.
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook
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
    
    # Try to improve with random search for lower ranks
    # The minimum rank is at least max(n, m, p) = 5, and at most n*m*p = 100
    # Let's try some ranks
    best_result = (U, V, W)
    best_rank = rank
    
    # Try ranks from 50 down to 60 first (moderate improvement)
    for r in range(min(80, rank), max(5, min(80, rank) - 25), -1):
        if time.time() > deadline - 15:
            break
        result = _try_random_rank(n, m, p, r, trials=50, timeout=min(5, deadline - time.time() - 15))
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = r
            break
    
    # If we found something better, try to go lower
    if best_rank < rank:
        for r in range(best_rank - 1, max(5, best_rank - 15), -1):
            if time.time() > deadline - 10:
                break
            result = _try_random_rank(n, m, p, r, trials=30, timeout=min(3, deadline - time.time() - 10))
            if result is not None:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = r
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
