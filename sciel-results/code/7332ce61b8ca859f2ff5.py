"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _total_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i],p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))

def _swap_delta(a, b, p, i, j):
    """Compute cost change when swapping facilities i and j in assignment p."""
    pi, pj = p[i], p[j]
    # Delta = sum_k [a[i,k]*(b[pj,pk] - b[pi,pk]) + a[j,k]*(b[pi,pk] - b[pj,pk])
    #         + a[k,i]*(b[pj,pk] - b[pi,pk]) + a[k,j]*(b[pi,pk] - b[pj,pk])]
    # Simplify: let d = b[pj,:] - b[pi,:]  (shape n)
    # Then delta = sum_k [a[i,k]*d[k] + a[j,k]*(-d[k]) + a[k,i]*d[k] + a[k,j]*(-d[k])]
    #            = sum_k [ (a[i,k] - a[j,k] + a[k,i] - a[k,j]) * d[k] ]
    d = b[pj] - b[pi]
    coeff = a[i] - a[j] + a[:, i] - a[:, j]
    return int(np.dot(coeff, d))

def _local_search_2opt(a, b, p, deadline):
    """2-opt local search: iteratively swap pairs that reduce cost."""
    n = len(p)
    p = p.copy()
    best_cost = _total_cost(a, b, p)
    
    # Precompute row sums of a for efficiency
    a_row_sums = a.sum(axis=1)
    
    improved = True
    while improved:
        improved = False
        # Try all pairs in a fixed order (or random order for diversity)
        for i in range(n):
            if time.time() > deadline:
                return p, best_cost
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    improved = True
                    # After swap, re-check this pair (might want to swap back or further)
                    # Continue scanning
    return p, best_cost

def _local_search_3opt(a, b, p, deadline):
    """3-opt local search: try 3-cycles (i,j,k) -> (j,k,i) etc."""
    n = len(p)
    p = p.copy()
    best_cost = _total_cost(a, b, p)
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, best_cost
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return p, best_cost
                for k in range(j + 1, n):
                    # Try cycle: p[i],p[j],p[k] -> p[j],p[k],p[i]
                    # This is equivalent to swapping i with j, then j with k
                    # But let's compute directly
                    pi, pj, pk = p[i], p[j], p[k]
                    # New assignment: i->pj, j->pk, k->pi
                    # Cost change involves all interactions
                    # Delta = cost(new) - cost(old) for the 3x3 subproblem + interactions with rest
                    # For efficiency, compute delta for the 3-cycle
                    # Old contributions involving i,j,k:
                    # sum_{x in {i,j,k}, y in all} a[x,y]*b[old_p[x], old_p[y]]
                    # New: sum_{x in {i,j,k}, y in all} a[x,y]*b[new_p[x], old_p[y]]
                    # plus the internal 3x3 changes
                    # This is complex; use simpler approach:
                    # Do it as two swaps: swap(i,j) then swap(j,k) in the new arrangement
                    # But that changes which elements are where.
                    
                    # Alternative: compute delta directly
                    # Let old_p = [pi, pj, pk] for positions i,j,k
                    # Let new_p = [pj, pk, pi] for positions i,j,k
                    # Delta = sum_{x in {i,j,k}, y in all} a[x,y]*(b[new_p[x], old_p[y]] - b[old_p[x], old_p[y]])
                    #      + sum_{x in all, y in {i,j,k}} a[x,y]*(b[old_p[x], new_p[y]] - b[old_p[x], old_p[y]])
                    #      - internal double count correction
                    # This gets messy. Let's just use 2-opt primarily.
                    pass
    return p, best_cost

def _eigenvector_order(a, b):
    """Create initial assignment using dominant eigenvectors."""
    n = len(a)
    # Symmetrize for eigenvalue computation
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Get dominant eigenvectors
    try:
        _, vecs_a = np.linalg.eigh(a_sym)
        _, vecs_b = np.linalg.eigh(b_sym)
    except:
        # Fallback to random
        return np.random.permutation(n)
    
    # Use the eigenvector with largest absolute eigenvalue
    # For a: largest magnitude eigenvalue
    eigvals_a = np.abs(np.linalg.eigvalsh(a_sym))
    eigvals_b = np.abs(np.linalg.eigvalsh(b_sym))
    
    idx_a = np.argmax(eigvals_a)
    idx_b = np.argmax(eigvals_b)
    
    v_a = vecs_a[:, idx_a]
    v_b = vecs_b[:, idx_b]
    
    # Sort facilities by their projection on v_a (descending)
    # Sort locations by their projection on v_b (ascending) - heavy to short
    fac_order = np.argsort(-v_a, kind='stable')
    loc_order = np.argsort(v_b, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    return p

def _random_solution(n):
    """Generate a random permutation."""
    return np.random.permutation(n)

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_p = None
    best_cost = np.inf
    
    # Generate diverse initial solutions
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            p = _eigenvector_order(a, b)
        elif restart == 1:
            # Rank-matching (original seed)
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif restart == 2:
            # Reverse rank-matching
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(-b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        else:
            p = _random_solution(n)
        
        # Local search
        p, cost = _local_search_2opt(a, b, p, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    if best_p is None:
        # Should not happen, but fallback
        best_p = _eigenvector_order(a, b)
        best_cost = _total_cost(a, b, best_p)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
