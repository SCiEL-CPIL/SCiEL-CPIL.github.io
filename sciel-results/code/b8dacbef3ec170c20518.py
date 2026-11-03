"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _compute_cost(a, b, p):
    """Compute the QAP cost for assignment p."""
    n = len(p)
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can compute this efficiently using matrix multiplication
    # Let B_p be the permutation of b: B_p[i,j] = b[p[i], p[j]]
    # Then cost = sum(a * B_p) = trace-like
    # Actually: sum_{i,j} a[i,j] * b[p[i], p[j]]
    # = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # We can do: for each i, sum_j a[i,j] * b[p[i], p[j]] = dot(a[i], b[p[i]])
    # So total = sum_i dot(a[i], b[p[i]])
    # This is: np.sum([np.dot(a[i], b[p[i]]) for i in range(n)])
    # Or more efficiently: np.einsum('ij,ij->', a, b[p])
    return int(np.einsum('ij,ij->', a, b[p], optimize=True))


def _compute_swap_delta(a, b, p, i, j):
    """Compute the change in cost if we swap p[i] and p[j]."""
    # Current contribution involving i and j:
    # Terms with i: sum_k a[i,k]*b[p[i],p[k]] + sum_k a[k,i]*b[p[k],p[i]] - a[i,i]*b[p[i],p[i]] (counted twice, subtract once)
    # Actually let's be more careful.
    # Total cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    # When we swap p[i] and p[j], let x = p[i], y = p[j].
    # New: p'[i]=y, p'[j]=x, p'[k]=p[k] for k!=i,j.
    # Delta = new - old
    # Only terms involving i or j change.
    # Terms where u=i or v=i or u=j or v=j are affected.
    
    # Let's compute the old and new contributions for all pairs (u,v) where u in {i,j} or v in {i,j}.
    # But that's 4n - 4 terms approximately. For n=150, that's ~600 operations per swap.
    
    # More efficient: 
    # old_cost = sum_{u,v} a[u,v]*b[p[u],p[v]]
    # new_cost = old_cost - (terms with p[i] or p[j] as location) + (terms with swapped)
    
    # Let's think of it as:
    # The contribution of facility i to the cost: sum_v a[i,v]*b[p[i],p[v]] + sum_u a[u,i]*b[p[u],p[i]] - a[i,i]*b[p[i],p[i]]
    # Wait, no. Each term a[u,v]*b[p[u],p[v]] is counted once. So the "row i" contribution is sum_v a[i,v]*b[p[i],p[v]] and the "column i" contribution is sum_u a[u,i]*b[p[u],p[i]]. But the term a[i,i]*b[p[i],p[i]] is in both.
    
    # Actually, let's just compute the delta directly by considering which b entries change.
    # b[p[i], *] and b[*, p[i]] and b[p[j], *] and b[*, p[j]] are replaced.
    
    # Let x = p[i], y = p[j]
    x = p[i]
    y = p[j]
    
    # Old terms involving location x (as first or second index in b):
    # sum_v a[i,v]*b[x, p[v]] + sum_u a[u,i]*b[p[u], x] - a[i,i]*b[x,x]  [because a[i,i]*b[x,x] counted in both]
    # Plus terms where v=j: a[i,j]*b[x, p[j]] = a[i,j]*b[x, y] is included in sum_v
    # And terms where u=j: a[j,i]*b[p[j], x] = a[j,i]*b[y, x] is included in sum_u
    
    # This is getting complex. Let me just compute it straightforwardly for n=150.
    # The number of affected terms is O(n). Let's do it.
    
    # Affected pairs: (i, v) for all v, (u, i) for all u, (j, v) for all v, (u, j) for all u
    # But (i,i), (i,j), (j,i), (j,j) are in multiple sets.
    
    # Let me just compute old and new for the affected submatrix.
    # Facilities i and j have their locations swapped.
    # All pairs (u,v) where u==i or v==i or u==j or v==j are affected.
    
    # Build the set of affected (u,v) pairs
    # For efficiency, iterate over u in {i,j} and all v, plus all u and v in {i,j}
    # That's: (i, all v), (j, all v), (all u, i), (all u, j)
    # Union: (i,v) for all v, (j,v) for all v, (u,i) for all u, (u,j) for all u
    
    # Let's compute:
    # old_part = sum over affected (u,v) of a[u,v]*b[p[u],p[v]]
    # new_part = sum over affected (u,v) of a[u,v]*b[p'[u],p'[v]]
    # where p'[i]=y, p'[j]=x, p'[u]=p[u] otherwise
    
    # To avoid double counting, let's define the affected set as:
    # {(u,v) : u in {i,j} or v in {i,j}}
    
    # We can compute this as:
    # row_i: (i, v) for v in 0..n-1
    # row_j: (j, v) for v in 0..n-1
    # col_i: (u, i) for u in 0..n-1, u != i (to avoid double count with row_i)
    # col_j: (u, j) for u in 0..n-1, u != j (to avoid double count with row_j)
    # But (i,i) is in row_i, (j,j) is in row_j, (i,j) is in row_i, (j,i) is in row_j
    # col_i excludes u=i, so (i,i) not in col_i. col_j excludes u=j, so (j,j) not in col_j.
    # What about (i,j) in col_j? u=i, v=j. i != j so it's included. But (i,j) is already in row_i. Double count!
    
    # Let me use a cleaner approach:
    # affected = {(u,v) : u==i or v==i or u==j or v==j}
    # = row_i ∪ row_j ∪ col_i ∪ col_j
    # Use inclusion-exclusion or just compute with a mask.
    
    # For n=150, let's just use numpy to compute it.
    # Create mask for affected
    # Actually, let's just compute it directly with loops since it's O(n) per swap and we'll do many swaps.
    # But 150*150 = 22500 potential swaps, and each is O(n)=150, so 3.375M operations per full pass. That's fine.
    
    # Let me compute:
    n = len(p)
    
    # Old contribution of affected terms
    old = 0
    # (i, v) for all v
    for v in range(n):
        old += a[i, v] * b[x, p[v]]
    # (u, i) for all u != i
    for u in range(n):
        if u != i:
            old += a[u, i] * b[p[u], x]
    # (j, v) for all v
    for v in range(n):
        old += a[j, v] * b[y, p[v]]
    # (u, j) for all u != j
    for u in range(n):
        if u != j:
            old += a[u, j] * b[p[u], y]
    # Subtract double-counted: (i,i), (i,j), (j,i), (j,j)
    # (i,i) is in row_i (v=i) and col_i (u=i, but we excluded u=i). So only once. OK.
    # (i,j) is in row_i (v=j) and col_j (u=i, i!=j so included). Double counted!
    # (j,i) is in row_j (v=i) and col_i (u=j, j!=i so included). Double counted!
    # (j,j) is in row_j (v=j) and col_j (u=j, but we excluded u=j). So only once. OK.
    # So subtract: a[i,j]*b[x,y] and a[j,i]*b[y,x]
    old -= a[i, j] * b[x, y]
    old -= a[j, i] * b[y, x]
    
    # New contribution
    new = 0
    # (i, v) for all v, but p'[i]=y
    for v in range(n):
        if v == i:
            new += a[i, i] * b[y, y]
        elif v == j:
            new += a[i, j] * b[y, x]
        else:
            new += a[i, v] * b[y, p[v]]
    # (u, i) for all u != i, p'[i]=y
    for u in range(n):
        if u == j:
            new += a[j, i] * b[x, y]
        else:
            new += a[u, i] * b[p[u], y]
    # (j, v) for all v, p'[j]=x
    for v in range(n):
        if v == i:
            new += a[j, i] * b[x, y]
        elif v == j:
            new += a[j, j] * b[x, x]
        else:
            new += a[j, v] * b[x, p[v]]
    # (u, j) for all u != j, p'[j]=x
    for u in range(n):
        if u == i:
            new += a[i, j] * b[y, x]
        else:
            new += a[u, j] * b[p[u], x]
    # Subtract double-counted: (i,j) and (j,i)
    # (i,j): in row_i (v=j) and col_j (u=i). Double counted.
    # (j,i): in row_j (v=i) and col_i (u=j). Double counted.
    new -= a[i, j] * b[y, x]
    new -= a[j, i] * b[x, y]
    
    return new - old


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    best_cost = _compute_cost(a, b, p)
    best_p = p[:]
    
    # 2-opt local search with random restarts
    import random
    random.seed(42)
    
    while time.time() < deadline:
        # Random perturbation: swap k random pairs
        k = random.randint(10, 30)
        p = best_p[:]
        for _ in range(k):
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
        
        # Local search: 2-opt
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all swaps
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    delta = _compute_swap_delta(a, b, p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        improved = True
                if time.time() > deadline:
                    break
        
        current_cost = _compute_cost(a, b, p)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p[:]
    
    return best_p
# EVOLVE-BLOCK-END
