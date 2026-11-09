"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
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

def _verify_fast(U, V, W):
    """
    Fast vectorized verification.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Convention: W index is k*n + i.
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    
    # Reshape to 3D for einsum
    # U[i, j, r]
    U3 = U.reshape(n, m, r)
    # V[j, k, r]
    V3 = V.reshape(m, p, r)
    # W[k, i, r] -- Note: W is stored as (p*n, r) with index k*n + i
    # So W_flat[k*n + i, r] corresponds to W3[k, i, r]
    W3 = W.reshape(p, n, r)
    
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    return np.allclose(T, 1.0, atol=1e-9)

def _try_dense_random_rank(target_rank, timeout=5, alphabet=(-2, 3)):
    """
    Attempt to find a random rank decomposition using denser random entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    attempts = 0
    lo, hi = alphabet
    
    while time.time() - start_time < timeout:
        attempts += 1
        # Use integers from a larger alphabet for denser matrices
        U = np.random.randint(lo, hi, size=(n * m, target_rank))
        V = np.random.randint(lo, hi, size=(m * p, target_rank))
        W = np.random.randint(lo, hi, size=(p * n, target_rank))
        
        if _verify_fast(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return None

def _try_sparse_random_rank(target_rank, timeout=5):
    """
    Attempt to find a random rank decomposition using sparse {-1, 0, 1} entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    attempts = 0
    
    while time.time() - start_time < timeout:
        attempts += 1
        U = np.random.randint(-1, 2, size=(n * m, target_rank))
        V = np.random.randint(-1, 2, size=(m * p, target_rank))
        W = np.random.randint(-1, 2, size=(p * n, target_rank))
        
        if _verify_fast(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return None

def _try_perturbed_schoolbook(target_rank, timeout=5):
    """
    Start from schoolbook structure and perturb columns to find lower rank.
    This is only applicable if target_rank >= 25 (the lower bound).
    We try to find a subset of columns from an expanded schoolbook-like structure.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    if target_rank < 25:
        return None
    
    # For ranks near the lower bound, try structured approaches
    # Create a base with more columns than needed, then select subsets
    # This is heuristic and may not be effective for 3x5x5
    
    # Alternative: try random but with correlation to schoolbook
    # For each of target_rank columns, randomly assign which (i,j,k) it corresponds to
    # but allow multiple (i,j,k) to share a column with appropriate signs
    
    while time.time() - start_time < timeout:
        # Generate random assignment of 75 tensor positions to target_rank columns
        assignments = np.random.randint(0, target_rank, size=75)
        
        U = np.zeros((n * m, target_rank), dtype=int)
        V = np.zeros((m * p, target_rank), dtype=int)
        W = np.zeros((p * n, target_rank), dtype=int)
        
        valid = True
        for idx in range(75):
            i = idx // (m * p)
            rem = idx % (m * p)
            j = rem // p
            k = rem % p
            
            col = assignments[idx]
            
            # Each column can contribute to at most one (i,j,k) with non-zero product
            # But we need sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for each (i,j,k)
            # This structured approach is complex; fall back to random if this fails
            
            # Simple approach: just set one column per (i,j,k) if possible
            # But we only have target_rank columns for 75 positions
            if target_rank < 75:
                valid = False
                break
        
        if not valid:
            continue
        
        # This structured approach is too constrained; use random instead
        break
    
    return None

def construct():
    deadline = time.time() + 110  # 10s margin for 120s limit
    
    # 1. Try schoolbook (guaranteed)
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # 2. Try to find lower rank
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25.
    # Known results suggest rank is between 25 and 40 for 3x5x5
    
    # Strategy: focus time on the most promising rank range
    # Use denser random entries which are more likely to find cancellations
    
    # Allocate time budget
    total_time = 110
    time_per_rank = 2.0  # seconds per rank attempt
    
    # Try ranks from 25 up to best_rank-1
    # Give more time to lower ranks (harder but more valuable)
    ranks_to_try = list(range(25, best_rank))
    
    # Weight time allocation: lower ranks get more time
    weights = []
    for r in ranks_to_try:
        # More time for lower ranks
        w = max(1.0, (best_rank - r) / 10.0)
        weights.append(w)
    
    total_weight = sum(weights)
    time_budget = total_time - 5  # reserve 5s
    
    # Try each rank with allocated time
    for idx, r in enumerate(ranks_to_try):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 1:
            break
        
        allocated = min(time_budget * weights[idx] / total_weight, remaining)
        if allocated < 0.5:
            break
        
        # Try dense random first (more likely to find solutions)
        res = _try_dense_random_rank(r, timeout=allocated * 0.7)
        if res:
            return res
        
        # If not found, try sparse random with remaining time
        remaining_alloc = allocated - (time.time() - (deadline - allocated))
        if remaining_alloc > 0.5:
            res = _try_sparse_random_rank(r, timeout=min(remaining_alloc, 2.0))
            if res:
                return res
        
    return best
# EVOLVE-BLOCK-END
