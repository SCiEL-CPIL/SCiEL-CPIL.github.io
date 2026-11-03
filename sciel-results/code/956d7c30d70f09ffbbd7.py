"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_cost_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping facilities at positions i and j
    in permutation p. p is an array where p[k] is the location assigned to
    facility k.
    
    We swap p[i] and p[j].
    Let x = p[i], y = p[j].
    New p has p[i]=y, p[j]=x.
    
    Cost change = sum_{k != i,j} a[i,k]*(b[y,p[k]] - b[x,p[k]]) 
                + sum_{k != i,j} a[j,k]*(b[x,p[k]] - b[y,p[k]])
                + a[i,j]*(b[y,x] - b[x,y])
                + a[j,i]*(b[x,y] - b[y,x])
                + a[i,i]*(b[y,y] - b[x,x])  # wait, this isn't right
    
    Let me re-derive. Cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    
    When we swap positions i and j (facilities i and j swap locations):
    p_new[i] = p[j], p_new[j] = p[i], p_new[k] = p[k] for k != i,j
    
    Delta = sum_{u,v} a[u,v] * (b[p_new[u], p_new[v]] - b[p[u], p[v]])
    
    Only terms where u or v is i or j change.
    
    Terms involving i or j:
    - u=i, v=i: a[i,i]*(b[p_new[i],p_new[i]] - b[p[i],p[i]]) = a[i,i]*(b[y,y]-b[x,x])
    - u=i, v=j: a[i,j]*(b[p_new[i],p_new[j]] - b[p[i],p[j]]) = a[i,j]*(b[y,x]-b[x,y])
    - u=j, v=i: a[j,i]*(b[p_new[j],p_new[i]] - b[p[j],p[i]]) = a[j,i]*(b[x,y]-b[y,x])
    - u=j, v=j: a[j,j]*(b[p_new[j],p_new[j]] - b[p[j],p[j]]) = a[j,j]*(b[x,x]-b[y,y])
    - u=i, v=k (k!=i,j): a[i,k]*(b[p_new[i],p_new[k]] - b[p[i],p[k]]) = a[i,k]*(b[y,p[k]]-b[x,p[k]])
    - u=j, v=k (k!=i,j): a[j,k]*(b[p_new[j],p_new[k]] - b[p[j],p[k]]) = a[j,k]*(b[x,p[k]]-b[y,p[k]])
    - u=k, v=i (k!=i,j): a[k,i]*(b[p_new[k],p_new[i]] - b[p[k],p[i]]) = a[k,i]*(b[p[k],y]-b[p[k],x])
    - u=k, v=j (k!=i,j): a[k,j]*(b[p_new[k],p_new[j]] - b[p[k],p[j]]) = a[k,j]*(b[p[k],x]-b[p[k],y])
    
    So:
    delta = a[i,i]*(b[y,y]-b[x,x]) + a[i,j]*(b[y,x]-b[x,y]) + a[j,i]*(b[x,y]-b[y,x]) + a[j,j]*(b[x,x]-b[y,y])
          + sum_{k!=i,j} a[i,k]*(b[y,p[k]]-b[x,p[k]])
          + sum_{k!=i,j} a[j,k]*(b[x,p[k]]-b[y,p[k]])
          + sum_{k!=i,j} a[k,i]*(b[p[k],y]-b[p[k],x])
          + sum_{k!=i,j} a[k,j]*(b[p[k],x]-b[p[k],y])
    
    This is O(n) per swap.
    """
    x = p[i]
    y = p[j]
    
    # Diagonal and cross terms
    delta = (a[i, i] * (b[y, y] - b[x, x])
             + a[i, j] * (b[y, x] - b[x, y])
             + a[j, i] * (b[x, y] - b[y, x])
             + a[j, j] * (b[x, x] - b[y, y]))
    
    # Sum over k != i, j
    # We can compute full sums and subtract the i,j terms
    # sum_{k} a[i,k]*(b[y,p[k]]-b[x,p[k]]) - terms where k=i or k=j
    
    # For k=i: a[i,i]*(b[y,p[i]]-b[x,p[i]]) = a[i,i]*(b[y,x]-b[x,x])
    # For k=j: a[i,j]*(b[y,p[j]]-b[x,p[j]]) = a[i,j]*(b[y,y]-b[x,y])
    
    # Similarly for other sums
    
    # Let's just do it with full arrays for clarity, it's still O(n)
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    # sum_{k in mask} a[i,k]*(b[y,p[k]]-b[x,p[k]])
    delta += np.sum(a[i, mask] * (b[y, p[mask]] - b[x, p[mask]]))
    # sum_{k in mask} a[j,k]*(b[x,p[k]]-b[y,p[k]])
    delta += np.sum(a[j, mask] * (b[x, p[mask]] - b[y, p[mask]]))
    # sum_{k in mask} a[k,i]*(b[p[k],y]-b[p[k],x])
    delta += np.sum(a[mask, i] * (b[p[mask], y] - b[p[mask], x]))
    # sum_{k in mask} a[k,j]*(b[p[k],x]-b[p[k],y])
    delta += np.sum(a[mask, j] * (b[p[mask], x] - b[p[mask], y]))
    
    return int(delta)


def _hungarian_init(a, b):
    """
    Use Hungarian algorithm to get a good initial assignment.
    We want to minimize sum a[i,j]*b[p[i],p[j]].
    
    One approach: match facility i to location j based on some criterion.
    A common heuristic: assign facility i to location j that minimizes 
    sum_k a[i,k] * b[j, loc(k)] but that's circular.
    
    Alternative: Use the fact that if we fix the assignment, the cost is
    sum_i sum_j a[i,j] b[p[i],p[j]]. 
    
    A simpler approach: sort facilities by row sum of a (descending) and 
    locations by row sum of b (ascending), then use Hungarian on a 
    reduced cost matrix.
    
    Actually, let's use a different approach: 
    1. Compute row sums of a: A_i = sum_j a[i,j]
    2. Compute row sums of b: B_j = sum_k b[j,k]
    3. We want to pair high A_i with low B_j.
    4. Create a cost matrix C[i,j] = A_i * B_j and use Hungarian to minimize it.
    """
    n = len(a)
    A = a.sum(axis=1)
    B = b.sum(axis=1)
    
    # Cost matrix for Hungarian: we want to minimize sum A[i] * B[p[i]]
    C = np.outer(A, B)
    
    # Hungarian algorithm minimizes the total cost
    _, col_ind = linear_sum_assignment(C)
    
    p = np.empty(n, dtype=np.int64)
    p[col_ind] = np.arange(n)
    
    return p


def _local_search(a, b, p, time_limit):
    """
    Perform local search with O(n) swap cost evaluation.
    """
    n = len(a)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Precompute some things for speed
    # We'll use first-improvement or best-improvement neighborhood
    
    # For speed, we'll do random swaps with delta evaluation
    rng = np.random.default_rng(42)
    
    improved = True
    no_improve_count = 0
    max_no_improve = 5000
    
    while improved and (time.time() - start_time) < deadline:
        improved = False
        no_improve_count = 0
        
        # Try all pairs in random order
        indices = np.arange(n)
        rng.shuffle(indices)
        
        for idx_i in indices:
            for idx_j in indices[idx_i + 1:]:
                if (time.time() - start_time) > deadline:
                    return p, current_cost
                
                delta = _swap_cost_delta(a, b, p, idx_i, idx_j)
                
                if delta < 0:
                    # Apply swap
                    p[idx_i], p[idx_j] = p[idx_j], p[idx_i]
                    current_cost += delta
                    improved = True
                    no_improve_count = 0
                    # After a swap, restart the inner loop for simplicity
                    # but for speed, we continue
                    break
            
            if improved and (time.time() - start_time) > deadline:
                return p, current_cost
            
            if improved:
                # Restart outer loop
                break
        
        if not improved:
            break
    
    return p, current_cost


def _simulated_annealing(a, b, p, time_limit, T0=10000.0, cooling=0.9999):
    """
    Simulated annealing with O(n) swap cost evaluation.
    """
    n = len(a)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    rng = np.random.default_rng(123)
    T = T0
    
    while (time.time() - start_time) < deadline:
        # Choose random swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        delta = _swap_cost_delta(a, b, p, i, j)
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            # Accept
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= cooling
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    total_time_limit = 150.0  # Safety margin: evaluator kills at 150s
    
    # Phase 1: Hungarian initialization
    p = _hungarian_init(a, b)
    current_cost = _qap_cost(a, b, p)
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed - 5.0  # 5s safety margin
    
    if remaining < 10.0:
        # Not enough time for search, return current
        return p.tolist()
    
    # Phase 2: Local search (first improvement)
    time_for_ls = min(remaining * 0.4, 60.0)
    p, current_cost = _local_search(a, b, p, time_for_ls)
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed - 5.0
    
    if remaining < 10.0:
        return p.tolist()
    
    # Phase 3: Simulated annealing for further improvement
    time_for_sa = remaining * 0.8
    p_sa, cost_sa = _simulated_annealing(a, b, p, time_for_sa)
    
    if cost_sa < current_cost:
        p = p_sa
        current_cost = cost_sa
    
    # Final local search polish
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed - 2.0
    
    if remaining > 5.0:
        p, current_cost = _local_search(a, b, p, remaining)
    
    return p.tolist()
# EVOLVE-BLOCK-END
