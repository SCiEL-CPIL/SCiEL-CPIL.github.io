"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_decode(adj, order):
    """Decode greedy MIS given a vertex ordering. Returns set of picked vertices."""
    n = adj.shape[0]
    picked = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    for v in order:
        if not picked[v] and not blocked[v]:
            picked[v] = True
            blocked[v] = True
            neighbors = np.where(adj[v] == 1)[0]
            blocked[neighbors] = True
    return picked


def _compute_core_numbers(adj):
    """Compute degeneracy ordering (core numbers) for each vertex."""
    n = adj.shape[0]
    degree = adj.sum(axis=1).copy()
    core = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    for _ in range(n):
        if not remaining.any():
            break
        # Find vertex with minimum degree among remaining
        degs = np.where(remaining, degree, n + 1)
        v = np.argmin(degs)
        if not remaining[v]:
            break
        core[v] = int(degree[v])
        remaining[v] = False
        neighbors = np.where(adj[v] == 1)[0]
        neighbors = neighbors[remaining[neighbors]]
        degree[neighbors] -= 1
    return core


def _baseline_scores(adj, core_numbers):
    """Generate baseline priority scores from structural features."""
    n = adj.shape[0]
    degree = adj.sum(axis=1).astype(float)
    
    # Compute neighborhood edge density
    A2 = adj @ adj
    two_edges_among_N = (adj * A2).sum(axis=1)
    edges_among_N = two_edges_among_N / 2.0
    max_possible = degree * (degree - 1) / 2.0
    max_possible_safe = np.where(max_possible > 0, max_possible, 1.0)
    neigh_density = np.where(max_possible > 0, edges_among_N / max_possible_safe, 0.0)
    
    # Cost: weighted sum of neighbor values
    neighbor_value = 1.0 / (1.0 + degree) * (1.0 - neigh_density)
    cost = adj @ neighbor_value
    
    # Normalize
    def _norm(x):
        mn, mx = x.min(), x.max()
        rng = mx - mn if mx > mn else 1.0
        return (x - mn) / rng
    
    deg_norm = _norm(degree)
    cost_norm = _norm(cost)
    core_norm = _norm(core_numbers.astype(float))
    
    score = -1.0 * deg_norm - 0.4 * cost_norm - 0.2 * core_norm
    return score


def _local_search_order(adj, initial_order, max_iters=200, time_limit=None):
    """Local search: try swapping adjacent pairs in the ordering, keep if greedy improves."""
    n = adj.shape[0]
    order = list(initial_order)
    
    # Initial evaluation
    picked = _greedy_decode(adj, order)
    best_size = picked.sum()
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time_limit is not None and time.time() - time_limit[0] > time_limit[1]:
            break
        
        for i in range(n - 1):
            if time_limit is not None and time.time() - time_limit[0] > time_limit[1]:
                break
            # Swap order[i] and order[i+1]
            order[i], order[i+1] = order[i+1], order[i]
            picked = _greedy_decode(adj, order)
            size = picked.sum()
            if size > best_size:
                best_size = size
                improved = True
            else:
                # Revert
                order[i], order[i+1] = order[i+1], order[i]
    
    return order, best_size


def _local_search_set(adj, initial_picked, max_iters=100, time_limit=None):
    """Local search on the independent set: try adding vertices, removing conflicts."""
    n = adj.shape[0]
    picked = set(np.where(initial_picked)[0].tolist())
    
    # Build adjacency lists for fast lookup
    adj_lists = [list(np.where(adj[i] == 1)[0]) for i in range(n)]
    
    best_size = len(picked)
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - time_limit[0] > time_limit[1]:
            break
        
        improved = False
        # Try adding each unpicked, unblocked vertex
        blocked = set()
        for v in picked:
            for u in adj_lists[v]:
                blocked.add(u)
        
        candidates = [v for v in range(n) if v not in picked and v not in blocked]
        
        for v in candidates:
            if time_limit is not None and time.time() - time_limit[0] > time_limit[1]:
                break
            new_picked = picked | {v}
            # Check independence
            valid = True
            for u in adj_lists[v]:
                if u in picked:
                    valid = False
                    break
            if valid and len(new_picked) > best_size:
                picked = new_picked
                best_size = len(new_picked)
                improved = True
                # Recompute blocked
                blocked = set()
                for v2 in picked:
                    for u in adj_lists[v2]:
                        blocked.add(u)
                break  # Restart candidate scan after improvement
    
    return picked, best_size


def priority(adjacency):
    """
    Assign a priority score to each vertex for the greedy MIS decoder.
    Uses local search to find a good vertex ordering, then derives scores.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Compute structural features
    core_numbers = _compute_core_numbers(adj)
    baseline = _baseline_scores(adj, core_numbers)
    
    # Get initial ordering from baseline scores
    initial_order = np.argsort(-baseline)
    
    # Time budget: leave 10s safety margin
    time_budget = 120.0 - 10.0  # 110s for search
    time_limit = [start_time, time_budget]
    
    # Multi-start local search on ordering
    best_order = list(initial_order)
    best_size = 0
    
    # First: local search from baseline ordering
    order, size = _local_search_order(adj, initial_order, max_iters=50, time_limit=time_limit)
    if size > best_size:
        best_size = size
        best_order = order
    
    # Try a few different initial orderings
    rng = np.random.RandomState(42)
    
    # Ordering 2: reverse degree
    degree = adj.sum(axis=1)
    order2 = np.argsort(-degree)  # High degree first (different strategy)
    if time.time() - start_time < start_time + time_budget * 0.6:
        order2_list, size2 = _local_search_order(adj, order2, max_iters=30, time_limit=time_limit)
        if size2 > best_size:
            best_size = size2
            best_order = order2_list
    
    # Ordering 3: core number based
    order3 = np.argsort(core_numbers)  # Low core number first
    if time.time() - start_time < start_time + time_budget * 0.8:
        order3_list, size3 = _local_search_order(adj, order3, max_iters=30, time_limit=time_limit)
        if size3 > best_size:
            best_size = size3
            best_order = order3_list
    
    # Ordering 4: random perturbation of baseline
    if time.time() - start_time < start_time + time_budget * 0.9:
        perm = rng.permutation(n)
        # Mix: 70% baseline order, 30% random
        mixed_order = list(initial_order)
        for i in range(0, n, 3):
            if i < n:
                mixed_order[i] = perm[i % n]
        mixed_list, size4 = _local_search_order(adj, mixed_order, max_iters=30, time_limit=time_limit)
        if size4 > best_size:
            best_size = size4
            best_order = mixed_list
    
    # Derive scores from best ordering: earlier in order = higher score
    # Use rank-based scores with small perturbations for ties
    score = np.zeros(n, dtype=float)
    rank_map = np.zeros(n, dtype=int)
    for pos, v in enumerate(best_order):
        rank_map[v] = pos
    
    # Score = n - rank (earlier = higher score)
    score = (n - rank_map).astype(float)
    
    # Add tiny noise for deterministic tie-breaking
    score += rng.uniform(0, 1e-6, size=n)
    
    return score

# EVOLVE-BLOCK-END
