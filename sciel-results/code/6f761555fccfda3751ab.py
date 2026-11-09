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
    """Build adjacency as bitmasks (Python ints) for O(1) neighbor operations."""
    val_set = set(values)
    n = len(values)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if values[i] + values[j] in val_set:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _color_bound(adj, P):
    """Greedy coloring upper bound on independence number of induced subgraph on P.
    Returns list of (color_class_mask) and total colors used.
    """
    # Order P by descending degree within P for tighter bound
    P_list = list(P)
    P_list.sort(key=lambda v: bin(adj[v] & P).count('1'), reverse=True)
    
    color_classes = []
    uncolored = P  # bitmask of vertices not yet colored
    while uncolored:
        # Greedy: pick a maximal independent set from uncolored
        cls = 0
        # Iterate through P_list order
        for v in P_list:
            if (uncolored >> v) & 1:
                # Check v is independent from current cls
                if not (adj[v] & cls):
                    cls |= (1 << v)
                    uncolored &= ~(1 << v)
        color_classes.append(cls)
    return len(color_classes)

def _branch_and_bound_mis(adj, n, initial_set, time_limit, rng, deadline):
    """Branch and bound for Maximum Independent Set.
    State: current independent set (bitmask), candidate set (bitmask).
    """
    best_size = bin(initial_set).count('1')
    best_set = initial_set
    start_time = time.time()
    
    # For branch and bound, we work with bitmasks
    # adj is a list of int bitmasks
    
    def _bb(P, I, depth):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        # Upper bound: |I| + |P|
        if bin(I).count('1') + bin(P).count('1') <= best_size:
            return
        
        if P == 0:
            sz = bin(I).count('1')
            if sz > best_size:
                best_size = sz
                best_set = I
            return
        
        # Color bound
        # For speed, if P is small, just use |P| as bound
        p_count = bin(P).count('1')
        if p_count <= 20:
            upper = bin(I).count('1') + p_count
        else:
            upper = bin(I).count('1') + _color_bound(adj, P)
        
        if upper <= best_size:
            return
        
        # Pick a vertex to branch on: highest degree in P
        # Find vertex in P with max degree
        max_deg = -1
        pivot = -1
        temp_p = P
        while temp_p:
            v = (temp_p & -temp_p).bit_length() - 1
            temp_p &= temp_p - 1
            deg = bin(adj[v] & P).count('1')
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot == -1:
            return
        
        # Branch 1: Include pivot
        new_I = I | (1 << pivot)
        # New candidates: P without pivot and without neighbors of pivot
        new_P = P & ~(1 << pivot) & ~adj[pivot]
        _bb(new_P, new_I, depth + 1)
        
        # Branch 2: Exclude pivot
        new_P2 = P & ~(1 << pivot)
        _bb(new_P2, I, depth + 1)
        
    _bb(initial_set, 0, 0)
    
    return best_set

def _solve_instance(values, time_budget, seed, deadline):
    """Solve one instance: find max sum-free subset using B&B."""
    n = len(values)
    if n == 0:
        return []
    
    rng = random.Random(seed)
    adj = _build_conflict_graph(values)
    
    best_size = 0
    best_mask = 0
    
    # Generate multiple initial solutions to seed B&B
    # 1. All vertices (then prune) - not valid, so use greedy
    # 2. Parity-based
    odds_mask = 0
    evens_mask = 0
    for i in range(n):
        if values[i] % 2 != 0:
            odds_mask |= (1 << i)
        else:
            evens_mask |= (1 << i)
    
    candidates = []
    candidates.append(odds_mask)
    candidates.append(evens_mask)
    
    # 3. Greedy by degree (low degree first)
    def greedy_mis_seed():
        remaining = (1 << n) - 1
        result = 0
        while remaining:
            # Find vertex with min degree in remaining
            min_deg = float('inf')
            v_pick = -1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = bin(adj[v] & remaining).count('1')
                if deg < min_deg:
                    min_deg = deg
                    v_pick = v
                    if deg == 0:
                        break
            result |= (1 << v_pick)
            remaining &= ~(1 << v_pick) & ~adj[v_pick]
        return result
    
    candidates.append(greedy_mis_seed())
    
    # 4. Random greedy
    for _ in range(3):
        remaining = (1 << n) - 1
        result = 0
        while remaining:
            # Pick random vertex from remaining
            idx = 0
            count = bin(remaining).count('1')
            pick_idx = rng.randint(0, count - 1)
            temp = remaining
            for _ in range(pick_idx + 1):
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if _ == pick_idx:
                    v_pick = v
            result |= (1 << v_pick)
            remaining &= ~(1 << v_pick) & ~adj[v_pick]
        candidates.append(result)
    
    # Evaluate candidates
    for mask in candidates:
        # Verify it's independent (should be by construction, but safe)
        # For speed, trust construction
        sz = bin(mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Run B&B with time limit
    # For n=60, B&B might be feasible with good bounds
    # For n=120, 240, B&B is likely intractable, so we rely on the greedy seeds
    # We'll run B&B only if n <= 80 and we have time
    if n <= 80 and time_budget > 1.0:
        bb_deadline = min(deadline, time.time() + time_budget)
        # Use the best candidate as initial
        best_set = _branch_and_bound_mis(adj, n, best_mask, time_budget, rng, bb_deadline)
        sz = bin(best_set).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = best_set
    elif n > 80:
        # For larger instances, do a local improvement pass
        # Try to add vertices from the complement if possible
        # This is a simple heuristic since B&B won't work
        current = best_mask
        remaining = ((1 << n) - 1) & ~current
        improved = True
        while improved and time.time() < deadline:
            improved = False
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if not (adj[v] & current):
                    current |= (1 << v)
                    remaining &= ~(1 << v)
                    improved = True
        best_mask = current
        best_size = bin(best_mask).count('1')
    
    # Convert mask to list of values
    result = [values[i] for i in range(n) if (best_mask >> i) & 1]
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s safety margin
    
    instances = [
        _instance(60, 801),
        _instance(120, 802),
        _instance(240, 803)
    ]
    
    results = []
    for i, values in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5  # 5s safety margin
        if remaining < 0:
            remaining = 0.1
        # Allocate more time to smaller instances where B&B can be exact
        if len(values) <= 80:
            time_budget = max(remaining * 0.5, 1.0)
        else:
            time_budget = max(remaining * 0.1, 0.5)
        
        subset = _solve_instance(values, time_budget, 801 + i, deadline)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
