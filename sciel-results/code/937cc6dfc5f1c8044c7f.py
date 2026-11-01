"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def _verify_factors(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    produce the matrix multiplication tensor under the trace convention W[k*n+i, r]."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Enumerate all triples (A, B, C)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                expected = 1
                actual = np.dot(U_arr[A], V_arr[B] * W_arr[C])
                if abs(actual - expected) > tol:
                    return False, (A, B, C, expected, actual)
    return True, None

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def _try_random_search(n, m, p, max_rank, trials=5000, seed=42, time_limit=100):
    """Random search for a low-rank decomposition."""
    rng = random.Random(seed)
    start_time = time.time()
    
    for trial in range(trials):
        if time.time() - start_time > time_limit:
            break
        
        rank = max_rank
        U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
        
        ok, _ = _verify_factors(U, V, W, n, m, p)
        if ok:
            return U, V, W, rank
    return None

def _structured_decomposition_attempt(n, m, p, target_rank=11):
    """Attempt a structured decomposition for M_{2,3,3} with rank < 18.
    
    For n=2, m=3, p=3, the schoolbook rank is 18. 
    We try to find a rank-11 or lower decomposition using algebraic structure.
    
    Known: Strassen's algorithm works for 2x2x2 (rank 7 vs 8).
    For M_{2,3,3}, we can try to exploit the rectangular structure.
    """
    # This is a heuristic attempt; if it fails, return None
    # A proper algebraic construction would be needed.
    # For now, return None to fall back to schoolbook.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120 - 5  # safety margin
    
    # First, try known lower-rank constructions if available
    # For 2x3x3, there may be a rank-11 decomposition but we need to verify.
    
    # Try a few structured approaches before falling back to schoolbook
    result = _structured_decomposition_attempt(n, m, p)
    if result is not None:
        U, V, W, rank = result
        ok, _ = _verify_factors(U, V, W, n, m, p)
        if ok:
            return {"U": U, "V": V, "W": W}
    
    # Try random search with decreasing rank
    for target_rank in [11, 12, 13, 14, 15, 16, 17]:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining_time = deadline - elapsed
        if remaining_time < 5:
            break
        
        result = _try_random_search(n, m, p, target_rank, trials=10000, seed=target_rank, time_limit=min(remaining_time - 5, 20))
        if result is not None:
            U, V, W, rank = result
            ok, _ = _verify_factors(U, V, W, n, m, p)
            if ok:
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
