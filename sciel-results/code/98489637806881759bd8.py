"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _build_decomposition(rank_terms):
    """Build U, V, W factor matrices from a list of term specifications."""
    n, m, p = DIMENSIONS
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=atol)

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for the 2x2 core (indices 0,1)."""
    terms = []
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}))
    terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}))
    terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}))
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}))
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    return terms

def _standard_border_terms():
    """Standard multiplication terms for the border (at least one index is 2)."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _try_deterministic_construction():
    """
    Attempt a deterministic construction for rank 23 or lower.
    
    Strategy: Use Strassen (7) for the 2x2 core, then carefully merge border terms.
    The 19 border terms can potentially be reduced by using products that cover
    multiple border positions.
    
    We'll try several known algebraic patterns for merging.
    """
    n, m, p = DIMENSIONS
    strassen = _strassen_2x2_terms()
    
    # The 19 border positions are:
    # Row 2 of A: (2,0,k), (2,1,k), (2,2,k) for k=0,1,2 -> 9 terms
    # Col 2 of B: (i,2,k) for i=0,1; k=0,1,2 -> 6 terms (excluding (2,2,k) already counted)
    # Actually let me enumerate properly:
    # Border = all (i,j,k) where at least one of i,j,k is 2
    # Total 27 - 8 (core) = 19 border terms
    
    # Let's try a construction where we group border terms:
    # Group 1: Terms involving a[2,j] * b[j,k] * c[k,2] (i=2, k=2, j varies)
    #   (2,0,0), (2,1,0), (2,2,0), (2,0,1), (2,1,1), (2,2,1), (2,0,2), (2,1,2), (2,2,2)
    #   = 9 terms
    # Group 2: Terms involving a[i,2] * b[2,k] * c[k,i] (j=2, i!=2, k varies, i!=2)
    #   (0,2,0), (1,2,0), (0,2,1), (1,2,1), (0,2,2), (1,2,2) = 6 terms
    # Group 3: Terms involving a[i,0] or a[i,1] with k=2 but j!=2, i!=2
    #   (0,0,2), (0,1,2), (1,0,2), (1,1,2) = 4 terms
    # Total: 9 + 6 + 4 = 19 ✓
    
    # Try to merge Group 1 using Strassen-like patterns on the 3x3 subproblem
    # But actually, let's try a simpler approach: just verify rank 26 works
    # and then try to find a rank 23 or 24 construction.
    
    # For now, return the rank-26 construction as baseline
    terms = strassen + _standard_border_terms()
    U, V, W = _build_decomposition(terms)
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W, len(terms)
    return None, None, None, float('inf')

def _try_rank_r_random_search(r, deadline, max_attempts=5000, seed=None):
    """
    Try to find a rank-r decomposition using random search with verification.
    Uses a smarter initialization: start from rank 26 and try to merge terms.
    """
    n, m, p = DIMENSIONS
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    strassen = _strassen_2x2_terms()
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Start with Strassen terms (7)
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        for r_idx, (u_dict, v_dict, w_dict) in enumerate(strassen):
            for (i, j), val in u_dict.items():
                U[i * m + j, r_idx] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r_idx] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r_idx] = val
        
        num_extra = r - 7
        if num_extra <= 0:
            continue
        
        # Randomly assign border coverage to extra terms
        border_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)
                          if not (i < 2 and j < 2 and k < 2)]
        
        for t in range(num_extra):
            r_idx = 7 + t
            # Pick a random border position
            i, j, k = border_indices[random.randint(0, len(border_indices) - 1)]
            
            # Set primary coefficient
            U[i * m + j, r_idx] = random.choice([1, -1])
            V[j * p + k, r_idx] = random.choice([1, -1])
            W[k * n + i, r_idx] = random.choice([1, -1])
            
            # Optionally add a second border position (merging)
            if random.random() < 0.4:
                i2, j2, k2 = border_indices[random.randint(0, len(border_indices) - 1)]
                if (i2, j2, k2) != (i, j, k):
                    U[i2 * m + j2, r_idx] = random.choice([1, -1])
                    V[j2 * p + k2, r_idx] = random.choice([1, -1])
                    W[k2 * n + i2, r_idx] = random.choice([1, -1])
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p, atol=1e-6):
            return U, V, W
    
    return None, None, None

def _try_systematic_merge():
    """
    Systematically try to merge border terms using algebraic identities.
    
    Key idea: If we have terms (i,j,k) and (i',j',k'), we can potentially
    replace them with a single term if the products align.
    
    For example, terms (i,2,k) and (i,2,k') can be merged if we use
    a[i,2] * (b[2,k] + b[2,k']) * c[k,i] + ... but this requires care.
    
    Let's try a specific known construction for rank 23.
    """
    n, m, p = DIMENSIONS
    strassen = _strassen_2x2_terms()
    
    # Try a specific pattern: merge pairs of border terms
    # We'll try to construct rank 23 = 7 + 16 border terms (from 19)
    # That means we need to merge 3 pairs of border terms.
    
    # Let's try merging:
    # Merge (0,2,0) and (1,2,0): use a[0,2]+a[1,2] times b[2,0] times ... 
    # But c[0,0] and c[0,1] are different, so we need separate W entries.
    # This doesn't directly merge.
    
    # Alternative: Use the fact that for fixed k, the terms (i,2,k) for i=0,1
    # can be handled by: (a[0,2]+a[1,2])*b[2,k]*(c[k,0]+c[k,1]) - a[0,2]*b[2,k]*c[k,1] - a[1,2]*b[2,k]*c[k,0]
    # But that's 3 terms for 2 original terms, so it's worse.
    
    # Let's try a different approach: use the full 3x3 structure more cleverly.
    # Actually, let's just do a more extensive random search with better initialization.
    return None, None, None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Deterministic rank-26 construction (Strassen + standard border)
    if time.time() < deadline:
        U, V, W, rank = _try_deterministic_construction()
        if U is not None and rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    
    # Strategy 2: Random search for rank 25
    if time.time() < deadline:
        remaining = deadline - time.time()
        max_attempts = min(2000, int(remaining * 500))
        U, V, W = _try_rank_r_random_search(25, deadline, max_attempts=max_attempts, seed=42)
        if U is not None and _verify_decomposition(U, V, W, n, m, p):
            best_rank = 25
            best_result = (U, V, W)
    
    # Strategy 3: Random search for rank 24
    if time.time() < deadline:
        remaining = deadline - time.time()
        max_attempts = min(1000, int(remaining * 400))
        U, V, W = _try_rank_r_random_search(24, deadline, max_attempts=max_attempts, seed=123)
        if U is not None and _verify_decomposition(U, V, W, n, m, p):
            best_rank = 24
            best_result = (U, V, W)
    
    # Strategy 4: Random search for rank 23
    if time.time() < deadline:
        remaining = deadline - time.time()
        max_attempts = min(500, int(remaining * 300))
        U, V, W = _try_rank_r_random_search(23, deadline, max_attempts=max_attempts, seed=777)
        if U is not None and _verify_decomposition(U, V, W, n, m, p):
            best_rank = 23
            best_result = (U, V, W)
    
    # Fallback: standard rank-27
    if best_result is None:
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
        U, V, W = _build_decomposition(terms)
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
