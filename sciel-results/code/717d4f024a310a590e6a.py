"""QAPLIB tho150 solver: local search with incremental cost updates."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _cost_full(a, b, p):
    """Exact cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int(np.einsum('ij,kl,il,jl->', a, b, p, p, optimize=True))


def _make_delta(a, b, p):
    """delta[i, j] = marginal cost of placing facility i at location j,
    given the current assignment p (excluding the i,i self term).
    delta[i, j] = sum_k (a[i,k] + a[k,i]) * b[j, p[k]] - a[i,i]*b[j,j]
    """
    n = len(p)
    A2 = a + a.T
    # M[i, j] = sum_k A2[i,k] * b[j, p[k]]  ->  b @ p-indexed  times A2
    # b[j, p[k]] for all j,k = b[:, p]  shape (n, n)
    Bp = b[:, p]                      # (n, n): Bp[j, k] = b[j, p[k]]
    M = A2 @ Bp                       # (n, n): M[i, j] = sum_k A2[i,k]*b[j,p[k]]
    # subtract self term a[i,i]*b[j,j]
    M = M - np.outer(np.diag(a), np.diag(b))
    return M


def _try_move(delta, a, b, p, i, j, current_cost):
    """Try relocating facility i to location j (j != p[i]).
    Returns (improved, new_cost, new_delta) or (False, current_cost, delta)."""
    n = len(p)
    old_j = p[i]
    if old_j == j:
        return False, current_cost, delta
    # cost delta for moving i from old_j to j
    d = delta[i, j] - delta[i, old_j]
    new_cost = current_cost + d
    if new_cost >= current_cost:
        return False, current_cost, delta
    # update p
    p[i] = j
    # update delta:
    # For all k != i, delta[k, j] changes because p[i] changed from old_j to j:
    #   delta[k, j] += A2[k,i]*b[j,j] - A2[k,i]*b[old_j, j]  ... wait
    # Actually delta[k, l] = sum_{m} A2[k,m]*b[l, p[m]] - a[k,k]*b[l,l]
    # When p[i] changes from old_j to j, only delta[k, l] for all k, l changes
    # through the term m=i: A2[k,i]*b[l, p[i]] becomes A2[k,i]*b[l, j] instead of A2[k,i]*b[l, old_j]
    # So delta[k, l] += A2[k,i]*(b[l, j] - b[l, old_j]) for all k, l
    # That's O(n^2). We can do it as: delta += outer(A2[:,i], (b[:,j] - b[:,old_j]))
    # But we also need to fix the self-term for row i (since p[i] changed, the self-term
    # for facility i at its new location j must be a[i,i]*b[j,j], which is already handled
    # by the formula, but the outer product update added A2[i,i]*(b[j,j]-b[j,old_j]) to delta[i,j]
    # and A2[i,i]*(b[j,old_j]-b[j,old_j])=0 to delta[i,old_j]. Let's just recompute row i.
    
    # Efficient: delta += outer(A2[:, i], b[:, j] - b[:, old_j])
    delta += np.outer(A2[:, i], b[:, j] - b[:, old_j])
    
    # Recompute row i exactly (the self-term correction)
    # delta[i, l] = sum_m A2[i,m]*b[l, p[m]] - a[i,i]*b[l,l]
    # We just updated all rows including i, but the self-term for row i at location l
    # should subtract a[i,i]*b[l,l]. The outer product update added A2[i,i]*b[l,j]-A2[i,i]*b[l,old_j]
    # to delta[i,l]. The correct self-term subtraction is a[i,i]*b[l,l] for all l.
    # Let's just recompute row i from scratch to be safe.
    Bp_new = b[:, p]
    delta[i, :] = A2[i, :] @ Bp_new - a[i, i] * np.diag(b)
    
    return True, new_cost, delta


def solve(a, b):
    t_start = time.time()
    deadline = t_start + 110.0  # 10s safety margin
    n = len(a)
    
    # Precompute
    A2 = a + a.T
    diag_a = np.diag(a)
    diag_b = np.diag(b)
    
    # --- Initial solution: rank-matching ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    current_cost = _cost_full(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    delta = _make_delta(a, b, p)
    
    # --- Local search loop ---
    rng = np.random.default_rng(42)
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Phase 1: single relocations (2-opt on one facility)
        # Iterate facilities in random order, try all target locations
        order = rng.permutation(n)
        for i in order:
            if time.time() > deadline - 5.0:
                break
            old_j = p[i]
            # Try all j != old_j
            # delta[i, :] gives marginal cost for each location
            # Find best j != old_j
            d = delta[i, :].copy()
            d[old_j] = np.inf
            j = int(np.argmin(d))
            if d[j] < 0:
                ok, new_cost, delta = _try_move(delta, a, b, p, i, j, current_cost)
                if ok:
                    current_cost = new_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        
        # Phase 2: pairwise 2-opt swaps
        if time.time() > deadline - 5.0:
            break
        swap_improved = True
        while swap_improved and time.time() < deadline - 5.0:
            swap_improved = False
            order = rng.permutation(n)
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                for j in order:
                    if i == j or p[i] == p[j]:
                        continue
                    if time.time() > deadline - 5.0:
                        break
                    # Cost of swapping facilities i and j
                    # delta_swap = (delta[i, p[j]] + delta[j, p[i]]) - (delta[i, p[i]] + delta[j, p[j]])
                    pi = p[i]
                    pj = p[j]
                    d_swap = (delta[i, pj] + delta[j, pi]) - (delta[i, pi] + delta[j, pj])
                    if d_swap < 0:
                        new_cost = current_cost + d_swap
                        # Perform swap
                        p[i], p[j] = pj, pi
                        # Update delta:
                        # p[i] changed from pi to pj, p[j] changed from pj to pi
                        # delta[k, l] += A2[k,i]*(b[l,pj]-b[l,pi]) + A2[k,j]*(b[l,pi]-b[l,pj])
                        # = (A2[k,i]-A2[k,j])*(b[l,pj]-b[l,pi])
                        diff_A = A2[:, i] - A2[:, j]
                        diff_b = b[:, pj] - b[:, pi]
                        delta += np.outer(diff_A, diff_b)
                        # Recompute rows i and j
                        Bp_new = b[:, p]
                        delta[i, :] = A2[i, :] @ Bp_new - diag_a[i] * diag_b
                        delta[j, :] = A2[j, :] @ Bp_new - diag_a[j] * diag_b
                        current_cost = new_cost
                        swap_improved = True
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
    
    # Final check: if we improved during the last iteration, do one more pass if time permits
    if time.time() < deadline - 2.0:
        # One final single-relocation pass
        order = rng.permutation(n)
        for i in order:
            if time.time() > deadline - 2.0:
                break
            old_j = p[i]
            d = delta[i, :].copy()
            d[old_j] = np.inf
            j = int(np.argmin(d))
            if d[j] < 0:
                ok, new_cost, delta = _try_move(delta, a, b, p, i, j, current_cost)
                if ok:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    # Ensure best_p is valid
    if len(best_p) != n or len(set(best_p.tolist())) != n:
        # Fallback: return identity
        return list(range(n))
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
