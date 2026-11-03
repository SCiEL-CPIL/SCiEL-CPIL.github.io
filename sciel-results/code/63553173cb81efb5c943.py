"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    # Compute sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficient: cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (sum_j a[i,j] * b[p[i], p[j]])
    # Let's compute using matrix operations:
    # cost = trace(a @ B) where B[i,j] = b[p[i], p[j]]
    # Actually: cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Let C = b[p][:, p]  (i.e., B[i,j] = b[p[i], p[j]])
    # Then cost = sum(a * C) = np.sum(a * C)
    B = b[np.ix_(p, p)]
    return np.sum(a * B)

def _swap_cost_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap positions i and j in permutation p.
    p[i] and p[j] are the locations currently assigned to facilities i and j.
    After swap: facility i goes to p[j], facility j goes to p[i].
    
    cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    
    When we swap p[i] <-> p[j], the affected terms are those involving index i or j.
    Let's denote old: pi = p[i], pj = p[j]
    New: p'[i] = pj, p'[j] = pi, p'[k] = p[k] for k != i,j
    
    Delta = new_cost - old_cost
    
    The terms that change are:
    - (i, i): a[i,i] * b[pj, pj] vs a[i,i] * b[pi, pi]
    - (j, j): a[j,j] * b[pi, pi] vs a[j,j] * b[pj, pj]
    - (i, j): a[i,j] * b[pj, pi] vs a[i,j] * b[pi, pj]
    - (j, i): a[j,i] * b[pi, pj] vs a[j,i] * b[pj, pi]
    - (i, k) for k != i,j: a[i,k] * b[pj, p[k]] vs a[i,k] * b[pi, p[k]]
    - (k, i) for k != i,j: a[k,i] * b[p[k], pj] vs a[k,i] * b[p[k], pi]
    - (j, k) for k != i,j: a[j,k] * b[pi, p[k]] vs a[j,k] * b[pj, p[k]]
    - (k, j) for k != i,j: a[k,j] * b[p[k], pi] vs a[k,j] * b[p[k], pj]
    
    Let's group:
    Terms with i (excluding j): 
      old: sum_k a[i,k]*b[pi,p[k]] + sum_k a[k,i]*b[p[k],pi]  (k != i,j)
           + a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj]
      new: sum_k a[i,k]*b[pj,p[k]] + sum_k a[k,i]*b[p[k],pj]  (k != i,j)
           + a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi]
    
    Similarly for j.
    
    This is getting complex. Let me use a simpler approach:
    delta = (new terms involving i or j) - (old terms involving i or j)
    
    Let S = set of all indices.
    Old contribution from rows/cols i and j:
      sum_{u in {i,j}, v in S} a[u,v] * b[p[u], p[v]]
      = sum_v a[i,v]*b[pi,p[v]] + sum_v a[j,v]*b[pj,p[v]]
    
    New contribution:
      sum_v a[i,v]*b[pj,p[v]] + sum_v a[j,v]*b[pi,p[v]]
      where p'[i]=pj, p'[j]=pi
    
    But wait, this double counts when v is in {i,j} and u is in {i,j}. Let's be careful.
    
    Actually, the full cost is sum_{u,v} a[u,v]*b[p[u],p[v]].
    When swapping i and j, only terms where u==i or u==j or v==i or v==j change.
    
    Let me just compute it directly for the affected terms.
    """
    pi = p[i]
    pj = p[j]
    
    # Old cost contributions involving i or j
    # Terms: (i,v) for all v, (v,i) for all v, (j,v) for all v, (v,j) for all v
    # But (i,i), (i,j), (j,i), (j,j) are counted twice if we do all four sums.
    # Let's compute:
    # old_affected = sum_v a[i,v]*b[pi,p[v]] + sum_v a[v,i]*b[p[v],pi] + sum_v a[j,v]*b[pj,p[v]] + sum_v a[v,j]*b[p[v],pj]
    # This counts (i,i) twice, (i,j) twice, (j,i) twice, (j,j) twice.
    # Actually no: (i,v) with v=i gives a[i,i]*b[pi,pi]. (v,i) with v=i gives a[i,i]*b[pi,pi]. So yes, double.
    
    # Let's just compute the delta by considering each affected (u,v) pair once.
    # Affected pairs: (i,v) for all v, (j,v) for all v, (v,i) for all v != i,j, (v,j) for all v != i,j
    # Wait, this is still messy. Let me use a different approach.
    
    # The change is:
    # For row i: a[i,:] * (b[pj, p] - b[pi, p])
    # For row j: a[j,:] * (b[pi, p] - b[pj, p])  
    # For col i (excluding row i,j): sum_{v not in {i,j}} a[v,i] * (b[p[v], pj] - b[p[v], pi])
    # For col j (excluding row i,j): sum_{v not in {i,j}} a[v,j] * (b[p[v], pi] - b[p[v], pj])
    # Plus diagonal and off-diagonal corrections...
    
    # This is getting too complex. Let me just recompute the full cost for small n, 
    # but for n=150 that's O(n^2) per swap which is 22500 ops, and we'd do many swaps.
    
    # Actually, let me precompute and use a smarter delta.
    # 
    # Let's define:
    # R_i = sum_v a[i,v] * b[pi, p[v]]  (row i contribution)
    # C_i = sum_v a[v,i] * b[p[v], pi]  (col i contribution)
    # 
    # When we swap i and j:
    # New R_i = sum_v a[i,v] * b[pj, p[v]]
    # New C_i = sum_v a[v,i] * b[p[v], pj]
    # New R_j = sum_v a[j,v] * b[pi, p[v]]
    # New C_j = sum_v a[v,j] * b[p[v], pi]
    #
    # Delta = (New R_i + New C_i + New R_j + New C_j) - (R_i + C_i + R_j + C_j)
    # But this double-counts (i,i), (i,j), (j,i), (j,j).
    
    # I'll just compute it naively but vectorized:
    # Old: 
    #   a[i,:] @ b[pi, p]  -> sum_v a[i,v]*b[pi,p[v]]
    #   a[:,i] @ b[p, pi]  -> sum_v a[v,i]*b[p[v],pi]
    #   a[j,:] @ b[pj, p]
    #   a[:,j] @ b[p, pj]
    # New:
    #   a[i,:] @ b[pj, p]
    #   a[:,i] @ b[p, pj]
    #   a[j,:] @ b[pi, p]
    #   a[:,j] @ b[p, pi]
    
    old_ri = np.dot(a[i, :], b[pi, p])
    old_ci = np.dot(a[:, i], b[p, pi])
    old_rj = np.dot(a[j, :], b[pj, p])
    old_cj = np.dot(a[:, j], b[p, pj])
    
    new_ri = np.dot(a[i, :], b[pj, p])
    new_ci = np.dot(a[:, i], b[p, pj])
    new_rj = np.dot(a[j, :], b[pi, p])
    new_cj = np.dot(a[:, j], b[p, pi])
    
    return (new_ri + new_ci + new_rj + new_cj) - (old_ri + old_ci + old_rj + old_cj)


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # 2-opt local search with random restarts
    rng = np.random.RandomState(42)
    
    while time.time() < deadline:
        # Try 2-opt swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Random order of pairs to avoid systematic bias
            for _ in range(n * n // 2):
                if time.time() >= deadline:
                    break
                i = rng.randint(n)
                j = rng.randint(n)
                if i == j:
                    continue
                delta = _swap_cost_delta(a, b, p, i, j)
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    best_p = p.copy()
                    improved = True
        
        # If we made improvements, try again from current state
        # Otherwise, do a random perturbation and restart
        if not improved:
            # Random perturbation: swap k random pairs
            k = rng.randint(2, 6)
            for _ in range(k):
                i = rng.randint(n)
                j = rng.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            # Now do local search again
            improved2 = True
            while improved2 and time.time() < deadline:
                improved2 = False
                for _ in range(n * n // 2):
                    if time.time() >= deadline:
                        break
                    i = rng.randint(n)
                    j = rng.randint(n)
                    if i == j:
                        continue
                    delta = _swap_cost_delta(a, b, p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        best_cost += delta
                        best_p = p.copy()
                        improved2 = True
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
