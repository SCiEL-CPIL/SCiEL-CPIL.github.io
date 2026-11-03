"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def compute_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
    return np.sum(a * b[np.ix_(p, p)])


def compute_swap_delta(a, b, p, i, j):
    """
    Compute the delta in cost when swapping facilities at positions i and j.
    p[i] and p[j] are the current locations.
    """
    x = p[i]
    y = p[j]
    
    # For all k != i, j:
    # Old: a[i,k]*b[x,p[k]] + a[j,k]*b[y,p[k]] + a[k,i]*b[p[k],x] + a[k,j]*b[p[k],y]
    # New: a[i,k]*b[y,p[k]] + a[j,k]*b[x,p[k]] + a[k,i]*b[p[k],y] + a[k,j]*b[p[k],x]
    # Delta for k: (a[i,k]-a[j,k])*(b[y,p[k]]-b[x,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],y]-b[p[k],x])
    
    # Build index mask excluding i and j
    n = len(p)
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    idx = np.where(mask)[0]
    
    pk = p[idx]
    
    delta = np.sum((a[i, idx] - a[j, idx]) * (b[y, pk] - b[x, pk]))
    delta += np.sum((a[idx, i] - a[idx, j]) * (b[pk, y] - b[pk, x]))
    
    # Handle k=i and k=j separately
    # When k=i: old has a[i,i]*b[x,x] + a[j,i]*b[y,x] + a[i,i]*b[x,x] + a[i,j]*b[x,y]
    #           new has a[i,i]*b[y,x] + a[j,i]*b[x,x] + a[i,i]*b[x,y] + a[i,j]*b[x,x]
    # When k=j: old has a[i,j]*b[x,y] + a[j,j]*b[y,y] + a[j,i]*b[x,y] + a[j,j]*b[y,y]
    #           new has a[i,j]*b[y,y] + a[j,j]*b[x,y] + a[j,i]*b[y,y] + a[j,j]*b[x,x]
    
    # k=i:
    old_ii = a[i,i]*b[x,x] + a[j,i]*b[y,x] + a[i,i]*b[x,x] + a[i,j]*b[x,y]
    new_ii = a[i,i]*b[y,x] + a[j,i]*b[x,x] + a[i,i]*b[x,y] + a[i,j]*b[x,x]
    delta += new_ii - old_ii
    
    # k=j:
    old_jj = a[i,j]*b[x,y] + a[j,j]*b[y,y] + a[j,i]*b[x,y] + a[j,j]*b[y,y]
    new_jj = a[i,j]*b[y,y] + a[j,j]*b[x,y] + a[j,i]*b[y,y] + a[j,j]*b[x,x]
    delta += new_jj - old_jj
    
    return delta


def local_search(a, b, p, time_limit, max_passes=100):
    """
    2-opt local search with first-improvement strategy.
    """
    n = len(p)
    p = p.copy()
    current_cost = compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    deadline = start_time + time_limit - 2.0  # 2 second safety margin
    
    pass_count = 0
    while pass_count < max_passes:
        if time.time() > deadline:
            break
        
        improved = False
        pass_count += 1
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                
                delta = compute_swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        
        if not improved:
            break
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin from 150s limit
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Local search
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        p, cost = local_search(a, b, p, remaining_time)
    
    return p.tolist()
# EVOLVE-BLOCK-END
