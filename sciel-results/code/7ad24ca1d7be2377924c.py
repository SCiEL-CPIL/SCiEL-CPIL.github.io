"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x:
            values.add(x)
    return sorted(values)

def _build_conflict_graph(values):
    """Build adjacency list for conflict graph.
    Edge (i,j) exists if there exists k such that values[i]+values[j]==values[k].
    For sum-free: we cannot have a+b=c where a,b,c all in subset.
    So vertex i conflicts with vertex j if values[i]+values[j] is also in values.
    """
    val_set = set(values)
    n = len(values)
    # adj[i] = set of j such that values[i] + values[j] in val_set
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            s = values[i] + values[j]
            if s in val_set:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def _is_sum_free(values, subset_indices):
    """Check if the subset is sum-free."""
    sub_set = set(values[i] for i in subset_indices)
    for i in subset_indices:
        for j in subset_indices:
            if values[i] + values[j] in sub_set:
                return False
    return True

def _greedy_mis(adj, n, rng):
    """Greedy Maximum Independent Set: repeatedly pick vertex with lowest degree in remaining graph."""
    in_set = [False] * n
    remaining = set(range(n))
    # Use a priority approach: pick vertex with min degree among remaining
    while remaining:
        # Find vertex with minimum degree in the induced subgraph on remaining
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
                if deg == 0:
                    break
        in_set[best_v] = True
        remaining.discard(best_v)
        # Remove neighbors of best_v from remaining
        for nb in list(adj[best_v]):
            if nb in remaining:
                remaining.discard(nb)
    return in_set

def _simulated_annealing(values, adj, n, rng, max_iters, t_start, t_end):
    """Simulated annealing on the conflict graph.
    State: boolean array in_set.
    Moves: flip one vertex (add or remove).
    Objective: maximize |in_set| subject to no conflicts within in_set.
    We track conflicts: number of edges where both endpoints are in_set.
    """
    # Start with greedy solution
    in_set = _greedy_mis(adj, n, rng)
    
    # Count current conflicts (should be 0 for a valid MIS)
    def count_conflicts(in_set):
        cnt = 0
        for i in range(n):
            if in_set[i]:
                for j in adj[i]:
                    if j > i and in_set[j]:
                        cnt += 1
        return cnt
    
    def objective(in_set):
        # We want to maximize size while keeping conflicts = 0
        # If conflicts > 0, it's invalid; penalize heavily
        size = sum(in_set)
        conf = count_conflicts(in_set)
        if conf > 0:
            return size - 1000 * conf
        return size
    
    cur_obj = objective(in_set)
    best_in_set = in_set[:]
    best_obj = cur_obj
    
    for it in range(max_iters):
        t = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Random move: flip a random vertex
        v = rng.randint(0, n - 1)
        in_set[v] = not in_set[v]
        
        new_obj = objective(in_set)
        delta = new_obj - cur_obj
        
        if delta >= 0 or rng.random() < pow(2.718281828, delta / max(t, 1e-10)):
            cur_obj = new_obj
            if cur_obj > best_obj:
                best_obj = cur_obj
                best_in_set = in_set[:]
        else:
            in_set[v] = not in_set[v]  # revert
    
    return best_in_set

def _solve_instance(values, time_budget, seed):
    """Solve one instance: find max sum-free subset."""
    n = len(values)
    if n == 0:
        return []
    
    rng = random.Random(seed)
    
    # Build conflict graph
    adj = _build_conflict_graph(values)
    
    # Try multiple strategies and keep the best
    best_subset = []
    best_size = 0
    
    # Strategy 1: Greedy MIS
    greedy = _greedy_mis(adj, n, rng)
    size = sum(greedy)
    if size > best_size:
        best_size = size
        best_subset = [values[i] for i in range(n) if greedy[i]]
    
    # Strategy 2: Simulated annealing with time budget
    elapsed = time.time()
    remaining_time = time_budget - (elapsed - _start_time)
    if remaining_time > 5:  # Need at least 5s for SA
        sa_iters = int(remaining_time * 3000)  # ~3000 iterations per second
        sa_iters = max(sa_iters, 1000)
        sa_iters = min(sa_iters, 200000)
        
        sa_result = _simulated_annealing(values, adj, n, rng, sa_iters, 10.0, 0.01)
        size = sum(sa_result)
        if size > best_size:
            best_size = size
            best_subset = [values[i] for i in range(n) if sa_result[i]]
    
    # Strategy 3: Try picking all odds/evens as baseline
    odds = [values[i] for i in range(n) if values[i] % 2 != 0]
    evens = [values[i] for i in range(n) if values[i] % 2 == 0]
    if len(odds) > best_size:
        best_size = len(odds)
        best_subset = odds
    if len(evens) > best_size:
        best_size = len(evens)
        best_subset = evens
    
    return best_subset

_start_time = None

def construct():
    global _start_time
    _start_time = time.time()
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803)
    ]
    
    results = []
    for i, values in enumerate(instances):
        # Allocate time proportionally but with minimum
        elapsed = time.time() - _start_time
        remaining = 120 - elapsed - 5  # 5s safety margin
        if remaining < 0:
            remaining = 0.1
        # Allocate more time to larger instances
        weight = len(values) / sum(len(v) for v in instances)
        time_budget = max(remaining * weight, 1.0)
        
        subset = _solve_instance(values, time_budget, 801 + i)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
