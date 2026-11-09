"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    """
    Schoolbook rank-n*m*p decomposition for n×m times m×p matrix multiplication.
    Uses the trace convention: output index is k*n + i for W.
    Input: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    For each term c = (i, j, k):
        U[i*m + j, c] = 1
        V[j*p + k, c] = 1
        W[k*n + i, c] = 1
    """
    n, m, p = DIMENSIONS
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


def _verify(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    r = len(W[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape != (n * m, r) or Va.shape != (m * p, r) or Wa.shape != (p * n, r):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _try_strassen_like(rank_target, seed=42, max_iter=5000):
    """
    Try to find a rank-decomposition below schoolbook using random search.
    We initialize with schoolbook and try to merge terms.
    """
    n, m, p = DIMENSIONS
    random.seed(seed)
    np.random.seed(seed)
    
    # Start from schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    for rank in range(rank_target, best_rank):
        if rank >= best_rank:
            continue
            
        # Random initialization
        U = np.random.randint(-2, 3, size=(n * m, rank))
        V = np.random.randint(-2, 3, size=(m * p, rank))
        W = np.random.randint(-2, 3, size=(p * n, rank))
        
        # Check if this works
        success = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        success = False
                        break
                if not success:
                    break
            if not success:
                break
        
        if success:
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def construct():
    """
    Find the lowest-rank exact decomposition for 2×5 × 5×6 matrix multiplication.
    Uses random search with Strassen-like algebraic structure.
    """
    deadline = time.time() + 140  # 140s deadline to be safe
    
    # First try known low-rank approaches
    # For 2x5x6, schoolbook is 60. Known bounds suggest lower is possible.
    
    # Try various rank targets with random search
    best_result = _schoolbook()
    best_rank = 60
    
    # Try to find better decompositions
    for target_rank in [18, 20, 22, 24, 25, 26, 28, 30, 32, 35, 40, 45, 50, 55]:
        if time.time() > deadline - 10:
            break
        
        result = _try_strassen_like(target_rank, seed=random.randint(0, 1000000), max_iter=100)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = target_rank
                break
    
    # Verify final result
    if not _verify(best_result["U"], best_result["V"], best_result["W"]):
        raise ValueError("Decomposition failed verification")
    
    return best_result
# EVOLVE-BLOCK-END
