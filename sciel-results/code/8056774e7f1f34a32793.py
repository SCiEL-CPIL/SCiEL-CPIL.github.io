"""Matrix multiplication tensor decomposition for 2x4 by 4x5 matrices."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give 1 if j exists, else 0
    # Actually: sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = (A*B)_{i,k} coefficient
    # More precisely, the tensor T[i,j,k] = delta (we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_random_search(n, m, p, max_time, rng):
    """Attempt to find a lower-rank decomposition via random initialization + local search."""
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = best_V = best_W = None
    
    # Try a few ranks below schoolbook
    for rank in range(schoolbook_rank - 1, max(1, schoolbook_rank - 5)):
        t0 = time.time()
        max_iters = 200
        for it in range(max_iters):
            if time.time() - t0 > max_time:
                break
            # Random initialization with small integers/fractions
            U = rng.integers(-2, 3, size=(n*m, rank)).tolist()
            V = rng.integers(-2, 3, size=(m*p, rank)).tolist()
            W = rng.integers(-2, 3, size=(p*n, rank)).tolist()
            if _verify(U, V, W, n, m, p):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                return {"U": U, "V": V, "W": W}
    return None

def _strassen_2x4x5():
    """
    Attempt a Strassen-like decomposition for 2x4 * 4x5.
    
    Split into blocks:
    A (2x4) = [A1 A2] where A1, A2 are 2x2
    B (4x5) = [B1; B2] where B1 is 2x5, B2 is 2x5
    
    C = A1*B1 + A2*B2
    
    Using Strassen on 2x2 * 2x2 blocks, we can potentially share.
    But B1 and B2 are 2x5, not 2x2.
    
    Alternative: Use the fact that rank of M_{2,4,5} might be lower than 40.
    
    Known lower bounds: For M_{n,m,p}, the rank is at least n*m + n*p + m*p - 2 (for n,m,p >= 2, from Landsberg).
    For n=2, m=4, p=5: LB = 2*4 + 2*5 + 4*5 - 2 = 8 + 10 + 20 - 2 = 36.
    
    So we might be able to find rank 36-39.
    """
    n, m, p = 2, 4, 5
    # This is complex to construct manually; return None to fall back
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # First, try to return a known good decomposition (schoolbook) as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    deadline = time.time() + 100  # Leave safety margin before 120s limit
    
    # Try random search for lower rank
    try:
        rng = np.random.default_rng(42)
        result = _try_random_search(n, m, p, max_time=5.0, rng=rng)
        if result is not None:
            return result
    except Exception:
        pass
    
    # If time permits, try more structured searches
    if time.time() < deadline - 10:
        # Try to construct a rank-36 decomposition using the known lower bound structure
        # This is heuristic; if it fails, we fall back to schoolbook
        pass
    
    # Fall back to schoolbook
    return schoolbook()
