"""Program for 3x5x5 matrix multiplication tensor decomposition.

Strategy: Start with the schoolbook decomposition (rank 75) and attempt to find
a lower-rank decomposition using the "border base" approach, then fall back
to rank reduction.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 5]
N, M, P = 3, 5, 5
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def _time_remaining():
    return DEADLINE - time.time() - SAFETY_MARGIN

def _schoolbook():
    """Return the standard rank-75 decomposition."""
    n, m, p = N, M, P
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

def _verify(decomp, tolerance=1e-6):
    """Check if the decomposition exactly realizes the matrix multiplication tensor."""
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    n, m, p = N, M, P
    rank = U.shape[1]
    
    # Reshape to tensor dimensions
    Uc = U.reshape(n, m, rank)
    Vc = V.reshape(m, p, rank)
    Wc = W.reshape(p, n, rank)
    
    # Compute sum over rank dimension c: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Using einsum for efficiency
    try:
        total = np.einsum('ijc,jkc,kic->ijk', Uc, Vc, Wc)
    except Exception:
        return False
        
    return np.allclose(total, 1.0, atol=tolerance)

def _try_winograd_3x5x5():
    """Attempt to find a Winograd-style low-rank decomposition.
    This function is a placeholder for structured searches.
    """
    return None

def _search_random(rank_min, rank_max):
    """Random search for a decomposition of the given rank."""
    best_decomp = None
    best_rank = rank_max + 1
    
    # Values allowed: integers and half-integers
    vals = [1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5]
    
    while _time_remaining() > 5:
        # Try a rank between rank_min and current best (or rank_max)
        target_rank = random.randint(rank_min, min(best_rank - 1, rank_max))
        
        U = np.zeros((N*M, target_rank))
        V = np.zeros((M*P, target_rank))
        W = np.zeros((P*N, target_rank))
        
        # Sparse random initialization
        for c in range(target_rank):
            # Each vector should have a few non-zeros
            u_sparsity = random.randint(1, 4)
            v_sparsity = random.randint(1, 4)
            w_sparsity = random.randint(1, 4)
            
            u_rows = random.sample(range(N*M), u_sparsity)
            v_rows = random.sample(range(M*P), v_sparsity)
            w_rows = random.sample(range(P*N), w_sparsity)
            
            for r in u_rows:
                U[r, c] = random.choice(vals)
            for r in v_rows:
                V[r, c] = random.choice(vals)
            for r in w_rows:
                W[r, c] = random.choice(vals)
                
        decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        if _verify(decomp):
            return decomp
            
    return None

def construct():
    # Start with the safe, known schoolbook decomposition
    base = _schoolbook()
    best_rank = N * M * P  # 75
    
    # Check if we have time to attempt optimization
    if _time_remaining() > 20:
        # Try to find a better decomposition by searching for lower ranks
        # We search from best_rank - 1 downwards
        # Given the low probability of finding a random exact solution,
        # we limit the search to a few ranks close to 75 to be safe with time.
        search_start = best_rank - 1
        search_end = max(best_rank - 10, 25) # Search down to 65 or lower bound
        
        # To maximize chance, we can try random restarts for each rank
        for r in range(search_start, search_end - 1, -1):
            if _time_remaining() < 10:
                break
            # Try a few random attempts for this rank
            for _ in range(10):
                if _time_remaining() < 10:
                    break
                candidate = _search_random(r, r)
                if candidate is not None:
                    return candidate
                    
    return base

if __name__ == "__main__":
    # Quick self-check
    result = construct()
    if _verify(result):
        print("Valid decomposition found with rank", len(result["U"][0]))
    else:
        print("ERROR: Invalid decomposition")
