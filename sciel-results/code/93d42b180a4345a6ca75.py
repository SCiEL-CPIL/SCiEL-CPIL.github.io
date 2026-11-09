"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _instance(size, seed, limit=10000):
    rng = random.Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x:
            values.add(x)
    return sorted(values)

def _build_conflict_graph(values):
    """Build adjacency as bitmasks for fast conflict counting."""
    val_set = set(values)
    n = len(values)
    # adj[i] is a bitmask of j such that values[i] + values[j] in val_set
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            if s in val_set:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _count_conflicts_bitmask(in_set, adj, n):
    """Count edges where both endpoints are in_set using bitmask AND."""
    mask = 0
    for i in range(n):
        if in_set[i]:
            mask |= (1 << i)
    cnt = 0
    for i in range(n):
        if in_set[i]:
            # Count bits in adj[i] & mask that are > i (to avoid double counting)
            common = adj[i] & mask
            # Only count j > i
            upper_mask = ((1 << n) - 1) ^ ((1 << (i + 1)) - 1)
            common_upper = common & upper_mask
            cnt += bin(common_upper).count('1')
    return cnt

def _greedy_mis_bitmask(adj, n, rng, order=None):
    """Greedy MIS using bitmask adjacency."""
    if order is None:
        order = list(range(n))
        rng.shuffle(order)
    
    in_set = [False] * n
    remaining_mask = (1 << n) - 1
    
    for v in order:
        if not (remaining_mask & (1 << v)):
            continue
        # Count degree in remaining
        deg = bin(adj[v] & remaining_mask).count('1')
        in_set[v] = True
        remaining_mask &= ~(1 << v)
        remaining_mask &= ~adj[v]
    
    return in_set

def _local_search_bitmask(in_set, adj, n, rng, max_iters, deadline):
    """Local search: try adding vertices or swapping to improve size while keeping sum-free."""
    cur_size = sum(in_set)
    best_in_set = in_set[:]
    best_size = cur_size
    cur_conf = 0  # Should be 0 for valid solution
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to add a random vertex not in set
        v = rng.randint(0, n - 1)
        if not in_set[v]:
            # Check if adding v creates conflicts
            new_conf = bin(adj[v] & ((1 << n) - 1)).count('1')
            # Count how many neighbors of v are in_set
            in_set_mask = 0
            for i in range(n):
                if in_set[i]:
                    in_set_mask |= (1 << i)
            conflicts_with_v = bin(adj[v] & in_set_mask).count('1')
            if conflicts_with_v == 0:
                in_set[v] = True
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_in_set = in_set[:]
        else:
            # Try removing v and adding two others? Complex. Just try swap.
            # Find a neighbor to swap
            neighbors_in_set = []
            in_set_mask = 0
            for i in range(n):
                if in_set[i]:
                    in_set_mask |= (1 << i)
            nb_mask = adj[v] & in_set_mask
            if nb_mask:
                # Get a random neighbor in set
                nb_list = [i for i in range(n) if (nb_mask & (1 << i))]
                if nb_list:
                    u = rng.choice(nb_list)
                    # Try swapping: remove u, add v
                    # But v is already in set... this is a no-op
                    pass
        
        # Alternative: try removing a random vertex and see if we can add more
        if rng.random() < 0.3:
            v = rng.randint(0, n - 1)
            if in_set[v]:
                in_set[v] = False
                cur_size -= 1
                # Try to add back some vertices that were blocked
                for w in range(n):
                    if not in_set[w]:
                        in_set_mask = 0
                        for i in range(n):
                            if in_set[i]:
                                in_set_mask |= (1 << i)
                        if bin(adj[w] & in_set_mask).count('1') == 0:
                            in_set[w] = True
                            cur_size += 1
                            if cur_size > best_size:
                                best_size = cur_size
                                best_in_set = in_set[:]
    
    return best_in_set

def _solve_instance(values, time_budget, seed, global_deadline):
    """Solve one instance: find max sum-free subset."""
    n = len(values)
    if n == 0:
        return []
    
    rng = random.Random(seed)
    
    # Build conflict graph as bitmasks
    adj = _build_conflict_graph(values)
    
    # Try multiple strategies and keep the best
    best_subset = []
    best_size = 0
    
    # Strategy 1: Multiple greedy restarts
    for restart in range(20):
        if time.time() > global_deadline:
            break
        greedy = _greedy_mis_bitmask(adj, n, rng)
        size = sum(greedy)
        if size > best_size:
            best_size = size
            best_subset = [values[i] for i in range(n) if greedy[i]]
    
    # Strategy 2: Local search from best greedy
    if best_subset and time.time() < global_deadline - 1.0:
        in_set = [False] * n
        for v in best_subset:
            idx = values.index(v)
            in_set[idx] = True
        
        ls_deadline = min(global_deadline - 0.5, time.time() + time_budget * 0.5)
        ls_result = _local_search_bitmask(in_set, adj, n, rng, 50000, ls_deadline)
        size = sum(ls_result)
        if size > best_size:
            best_size = size
            best_subset = [values[i] for i in range(n) if ls_result[i]]
    
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
    global_deadline = _start_time + 140  # 10s safety margin before 150s kill
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803)
    ]
    
    total_size = sum(len(v) for v in instances)
    results = []
    for i, values in enumerate(instances):
        elapsed = time.time() - _start_time
        remaining = global_deadline - elapsed
        if remaining < 0.1:
            remaining = 0.1
        
        # Allocate more time to larger instances
        weight = len(values) / total_size
        time_budget = max(remaining * weight * 0.8, 0.5)  # Use 80% of allocated, keep 20% as buffer
        
        subset = _solve_instance(values, time_budget, 801 + i, global_deadline)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
