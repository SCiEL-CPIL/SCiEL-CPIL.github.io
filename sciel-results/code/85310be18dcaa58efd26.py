# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Compute the exact cost change when swapping locations p[i] and p[j]."""
    pi, pj = p[i], p[j]
    # For all k != i, j: change is (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    # Plus the i-j interaction terms
    if i == j:
        return 0
    
    # Vectorized computation for k != i, j
    # We compute sum over all k, then subtract the i and j terms
    dA = a[i, :] - a[j, :]
    dB = b[pj, p] - b[pi, p]
    
    # Full dot product includes k=i and k=j terms
    full_dot = np.dot(dA, dB)
    
    # Subtract the k=i term: dA[i] * dB[i] = (a[i,i] - a[j,i]) * (b[pj, pi] - b[pi, pi])
    term_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    
    # Subtract the k=j term: dA[j] * dB[j] = (a[i,j] - a[j,j]) * (b[pj, pj] - b[pi, pj])
    term_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    # The k=i and k=j terms in the full dot are NOT the actual changes for those k
    # When k=i: the term a[i,i]*b[pi,pi] becomes a[i,i]*b[pj,pi], so change is a[i,i]*(b[pj,pi]-b[pi,pi])
    # When k=j: the term a[i,j]*b[pi,pj] becomes a[i,j]*b[pj,pj], so change is a[i,j]*(b[pj,pj]-b[pi,pj])
    # For facility j when k=i: a[j,i]*b[pj,pi] becomes a[j,i]*b[pi,pi], change is a[j,i]*(b[pi,pi]-b[pj,pi])
    # For facility j when k=j: a[j,j]*b[pj,pj] becomes a[j,j]*b[pi,pj], change is a[j,j]*(b[pi,pj]-b[pj,pj])
    
    # So the actual delta is:
    # sum_{k!=i,j} (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]]) 
    # + a[i,i]*(b[pj,pi]-b[pi,pi]) + a[i,j]*(b[pj,pj]-b[pi,pj])
    # + a[j,i]*(b[pi,pi]-b[pj,pi]) + a[j,j]*(b[pi,pj]-b[pj,pj])
    
    # The full_dot = sum_k (a[i,k]-a[j,k])*(b[pj,p[k]]-b[pi,p[k]])
    # = sum_{k!=i,j} ... + (a[i,i]-a[j,i])*(b[pj,pi]-b[pi,pi]) + (a[i,j]-a[j,j])*(b[pj,pj]-b[pi,pj])
    
    # So sum_{k!=i,j} = full_dot - term_i - term_j
    
    sum_others = full_dot - term_i - term_j
    
    # Now add the correct i,j interaction terms
    delta_ij = a[i, i] * (b[pj, pi] - b[pi, pi]) + a[i, j] * (b[pj, pj] - b[pi, pj]) \
             + a[j, i] * (b[pi, pi] - b[pj, pi]) + a[j, j] * (b[pi, pj] - b[pj, pj])
    
    return int(sum_others + delta_ij)


def _refine(a, b, p, deadline, best_cost, best_p):
    """First-improvement 2-opt (swap) local search until deadline.
    p is a numpy int64 array (a copy).
    Returns (best_cost, best_p) found so far."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    # Precompute for speed
    a_rows = a  # just reference
    b_rows = b
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += delta
                    improved = True
                    
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    
                    # Time check every few swaps
                    if (i * n + j) % 500 == 0 and time.time() >= deadline:
                        return best_cost, best_p
            if not improved and time.time() >= deadline:
                break
        if not improved:
            break

    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    return best_cost, best_p


def _anneal(a, b, p, start_time, deadline, best_cost, best_p, rng):
    """Simulated Annealing refinement.
    Returns (best_cost, best_p)."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    T = 10000.0
    T_end = 1.0
    alpha = 0.998
    
    steps = 0
    while T > T_end and (time.time() - start_time) < deadline:
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
            
        delta = _swap_delta(a, b, p, i, j)
        
        # Acceptance criterion
        if delta < 0 or (rng.random() < np.exp(-delta / T)):
            p[i], p[j] = p[j], p[i]
            cur += delta
            
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
                
        T *= alpha
        steps += 1
        
        # Periodic time check
        if steps % 100 == 0 and time.time() - start_time > deadline:
            break
            
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
        
    return best_cost, best_p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s kill

    rng = np.random.default_rng(42)
    
    best_cost = np.iinfo(np.int64).max
    best_p = None

    # --- Multiple diverse initial solutions ---
    # Strategy 1: Rank matching (heavy facilities to central locations)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    
    p = p0.copy()
    cur_cost = _qap_cost(a, b, p)
    best_cost, best_p = cur_cost, p.copy()
    
    # Refine
    p = p0.copy()
    best_cost, best_p = _refine(a, b, p, deadline, best_cost, best_p)
    
    # --- Strategy 2: Reverse rank matching ---
    if time.time() < deadline - 30:
        heavy2 = np.argsort(a.sum(axis=1), kind="stable")  # light facilities
        central2 = np.argsort(-b.sum(axis=1), kind="stable")  # peripheral locations
        p1 = np.empty(n, dtype=np.int64)
        p1[heavy2] = central2
        p = p1.copy()
        best_cost, best_p = _refine(a, b, p, deadline, best_cost, best_p)
    
    # --- Simulated Annealing from best ---
    if time.time() < deadline - 10:
        p = best_p.copy()
        best_cost, best_p = _anneal(a, b, p, t0, deadline, best_cost, best_p, rng)
    
    # --- Random restarts with local search ---
    restarts = 0
    while time.time() < deadline - 5.0:
        p = rng.permutation(n).astype(np.int64)
        best_cost, best_p = _refine(a, b, p, deadline, best_cost, best_p)
        restarts += 1
        if restarts >= 5:  # Limit restarts to avoid too many
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
