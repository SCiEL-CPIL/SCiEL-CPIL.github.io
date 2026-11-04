"""QAP wil100 solver with 2-opt + 3-opt cyclic exchange local search."""
import numpy as np
import time


def _total_cost(a, b, p):
    """Compute sum_{i,j} a[i,j] * b[p[i],p[j]]."""
    # p is array of facility indices at each location
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # = sum_i sum_j a[i,j] * b[p[i],p[j]]
    # Let Bp = b[p[:,None], p[None,:]]  (n x n)
    # cost = sum(a * Bp)
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_2opt(a, b, p, i, j):
    """Delta for swapping facilities at positions i and j."""
    fi = p[i]
    fj = p[j]
    # Current terms involving i or j:
    # sum_x a[x,i]*b[p[x],p[i]] + sum_x a[i,x]*b[p[i],p[x]] + (i,j terms counted once each)
    # After swap: p[i]<->p[j]
    # Delta = new - old for all pairs where at least one index is i or j
    
    # Efficient: the only pairs that change are (i,x), (x,i), (j,x), (x,j) for all x,
    # but (i,j) and (j,i) are shared.
    # Let's compute: for each x != i,j, terms a[i,x]*b[fi,p[x]] + a[x,i]*b[p[x],fi]
    # become a[i,x]*b[fj,p[x]] + a[x,i]*b[p[x],fj]
    # Similarly for j.
    # And the (i,j) pair: a[i,j]*b[fi,fj] + a[j,i]*b[fj,fi] becomes a[i,j]*b[fj,fi] + a[j,i]*b[fi,fj]
    
    # Vectorized:
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    px = p[mask]
    
    # Terms for i with others
    old_i = a[i, mask] * b[fi, px] + a[mask, i] * b[px, fi]
    new_i = a[i, mask] * b[fj, px] + a[mask, i] * b[px, fj]
    
    # Terms for j with others
    old_j = a[j, mask] * b[fj, px] + a[mask, j] * b[px, fj]
    new_j = a[j, mask] * b[fi, px] + a[mask, j] * b[px, fi]
    
    # Terms for (i,j) and (j,i)
    old_ij = a[i, j] * b[fi, fj] + a[j, i] * b[fj, fi]
    new_ij = a[i, j] * b[fj, fi] + a[j, i] * b[fi, fj]
    
    delta = (new_i + new_j + new_ij) - (old_i + old_j + old_ij)
    return int(delta)


def _delta_3opt_cyclic(a, b, p, i, j, k, direction=1):
    """
    Delta for 3-opt cyclic exchange.
    direction=1: facility at pos i -> pos j, pos j -> pos k, pos k -> pos i
    direction=-1: facility at pos i -> pos k, pos k -> pos j, pos j -> pos i
    """
    fi = p[i]
    fj = p[j]
    fk = p[k]
    
    if direction == 1:
        # After: pos i gets fk, pos j gets fi, pos k gets fj
        new_pi = fk
        new_pj = fi
        new_pk = fj
    else:
        # After: pos i gets fj, pos j gets fk, pos k gets fi
        new_pi = fj
        new_pj = fk
        new_pk = fi
    
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    mask[k] = False
    px = p[mask]
    
    # For each of the three positions, compute old and new contributions with all other positions
    # Position i: old facility fi, new facility new_pi
    old_i = a[i, mask] * b[fi, px] + a[mask, i] * b[px, fi]
    new_i = a[i, mask] * b[new_pi, px] + a[mask, i] * b[px, new_pi]
    
    # Position j: old facility fj, new facility new_pj
    old_j = a[j, mask] * b[fj, px] + a[mask, j] * b[px, fj]
    new_j = a[j, mask] * b[new_pj, px] + a[mask, j] * b[px, new_pj]
    
    # Position k: old facility fk, new facility new_pk
    old_k = a[k, mask] * b[fk, px] + a[mask, k] * b[px, fk]
    new_k = a[k, mask] * b[new_pk, px] + a[mask, k] * b[px, new_pk]
    
    # Pairs among i, j, k (6 ordered pairs)
    old_ijk = (a[i,j]*b[fi,fj] + a[j,i]*b[fj,fi] +
               a[i,k]*b[fi,fk] + a[k,i]*b[fk,fi] +
               a[j,k]*b[fj,fk] + a[k,j]*b[fk,fj])
    new_ijk = (a[i,j]*b[new_pi,new_pj] + a[j,i]*b[new_pj,new_pi] +
               a[i,k]*b[new_pi,new_pk] + a[k,i]*b[new_pk,new_pi] +
               a[j,k]*b[new_pj,new_pk] + a[k,j]*b[new_pk,new_pj])
    
    delta = (new_i + new_j + new_k + new_ijk) - (old_i + old_j + old_k + old_ijk)
    return int(delta)


def _polish_2opt(a, b, p, deadline):
    """First-improvement 2-opt local search with deadline."""
    n = len(p)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                d = _delta_2opt(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    improved = True
                    break
            if improved:
                break
    return p


def _polish_3opt(a, b, p, deadline):
    """First-improvement 3-opt cyclic exchange local search with deadline."""
    n = len(p)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                for k in range(j + 1, n):
                    for direction in (1, -1):
                        d = _delta_3opt_cyclic(a, b, p, i, j, k, direction)
                        if d < 0:
                            if direction == 1:
                                # i->j->k->i
                                p[i], p[j], p[k] = p[k], p[i], p[j]
                            else:
                                # i->k->j->i
                                p[i], p[j], p[k] = p[j], p[k], p[i]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return p


def solve(a, b):
    start_time = time.time()
    # Safety margin: stop searching 5s before the 120s limit (but we have 150s wall-clock)
    deadline = start_time + 110.0
    
    n = len(a)
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Polish with 2-opt first (fast convergence)
    p = _polish_2opt(a, b, p, deadline)
    
    # Then alternate 3-opt and 2-opt to escape local optima
    best_cost = _total_cost(a, b, p)
    best_p = p.copy()
    
    # Iterate: 3-opt then 2-opt, keeping best
    while time.time() < deadline:
        # 3-opt pass
        p = _polish_3opt(a, b, p, deadline)
        cost = _total_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        
        if time.time() >= deadline:
            break
        
        # 2-opt polish after 3-opt
        p = _polish_2opt(a, b, p, deadline)
        cost = _total_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    return best_p.tolist()
