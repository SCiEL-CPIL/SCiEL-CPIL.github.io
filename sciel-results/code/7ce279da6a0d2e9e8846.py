"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _is_sidon_k(subset, k):
    """Check if a subset has all pair sums (i<=j) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(candidates):
    """Find the largest Sidon subset size of the given 12-element set."""
    for size in range(12, 0, -1):
        for combo in itertools.combinations(candidates, size):
            if _is_sidon_k(combo, size):
                return size
    return 0

def _has_sidon3(arr):
    """Check if any 3-subset of arr is Sidon. Returns True if exists."""
    for sub3 in itertools.combinations(arr, 3):
        a, b, c = sub3
        # Check all 6 pair sums (with diagonals)
        sums = [a+a, a+b, a+c, b+b, b+c, c+c]
        if len(set(sums)) == 6:
            return True
    return False

def _has_sidon4(arr):
    """Check if any 4-subset of arr is Sidon. Returns True if exists."""
    for sub4 in itertools.combinations(arr, 4):
        if _is_sidon_k(sub4, 4):
            return True
    return False

def _collision_graph_density(values):
    """Compute density of the sum-collision graph.
    Edge (i,j) exists if values[i]+values[j] collides with another pair sum.
    Returns (density, adjacency matrix)."""
    n = len(values)
    sum_counts = {}
    for i in range(n):
        for j in range(i, n):
            s = values[i] + values[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
    
    adj = np.zeros((n, n), dtype=int)
    edge_count = 0
    for i in range(n):
        for j in range(i, n):
            s = values[i] + values[j]
            collision_count = sum_counts[s] - 1
            if collision_count > 0:
                adj[i, j] = collision_count
                adj[j, i] = collision_count
                edge_count += 1
    
    max_edges = n * (n - 1) // 2
    density = edge_count / max_edges if max_edges > 0 else 0
    return density, adj

def _min_vertex_cover_approx(adj):
    """Approximate minimum vertex cover size using greedy matching + LP relaxation.
    Returns lower bound on min vertex cover = n - max independent set."""
    n = len(adj)
    # Greedy maximum matching
    matched = [False] * n
    matching = []
    for i in range(n):
        if matched[i]:
            continue
        for j in range(i+1, n):
            if not matched[j] and adj[i, j] > 0:
                matched[i] = True
                matched[j] = True
                matching.append((i, j))
                break
    
    # For bipartite-like structure, min vertex cover >= max matching
    # Use greedy: pick vertices covering most edges
    remaining_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                remaining_edges.add((i, j))
    
    vc = set()
    while remaining_edges:
        # Find vertex covering most remaining edges
        best_v = -1
        best_count = -1
        for v in range(n):
            if v in vc:
                continue
            count = sum(1 for (i,j) in remaining_edges if i == v or j == v)
            if count > best_count:
                best_count = count
                best_v = v
        if best_v == -1:
            break
        vc.add(best_v)
        remaining_edges = {(i,j) for (i,j) in remaining_edges if i != best_v and j != best_v}
    
    return len(vc)

def _independence_number_upper_bound(adj):
    """Upper bound on independence number = n - lower bound on min vertex cover."""
    n = len(adj)
    vc_size = _min_vertex_cover_approx(adj)
    return n - vc_size

def _local_search_collision_graph(current_set, deadline, best_size, best_set, max_iters=5000):
    """Local search optimizing collision graph density."""
    current_set = list(current_set)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        density, adj = _collision_graph_density(current_set)
        
        # Find element with lowest degree in collision graph
        degrees = adj.sum(axis=1)
        min_idx = np.argmin(degrees)
        
        improved = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            new_val = random.randint(0, 128)
            if new_val in current_set:
                continue
            
            neighbor = current_set[:]
            neighbor[min_idx] = new_val
            neighbor.sort()
            
            # Quick check: does this neighbor have a 3-Sidon?
            if _has_sidon3(neighbor):
                # Check 4-Sidon only if no 3-Sidon was found... wait, we want NO Sidon subsets
                # Actually we want to MINIMIZE the max Sidon subset size
                # So we prefer sets where even 3-element subsets are NOT Sidon
                pass
            
            new_density, new_adj = _collision_graph_density(neighbor)
            
            if new_density > density:
                # Check actual max Sidon size
                ms = _max_sidon_size(neighbor)
                if ms < best_size:
                    best_size = ms
                    best_set = neighbor[:]
                    if best_size <= 2:
                        return best_set, best_size, True
                current_set = neighbor
                density = new_density
                improved = True
                break
        
        if not improved:
            # Try multiple random replacements
            idx = random.randint(0, 11)
            for _ in range(20):
                if time.time() > deadline:
                    break
                new_val = random.randint(0, 128)
                if new_val in current_set:
                    continue
                neighbor = current_set[:]
                neighbor[idx] = new_val
                neighbor.sort()
                
                new_density, new_adj = _collision_graph_density(neighbor)
                if new_density > density:
                    ms = _max_sidon_size(neighbor)
                    if ms < best_size:
                        best_size = ms
                        best_set = neighbor[:]
                        if best_size <= 2:
                            return best_set, best_size, True
                    current_set = neighbor
                    density = new_density
                    break
    
    return best_set, best_size, False

def _construct_modular_set(mod, offset=0):
    """Construct a set with strong modular structure to force collisions."""
    # Use values that are multiples or near multiples of small numbers
    # to create many sum collisions
    candidates = []
    for i in range(12):
        val = (i * mod + offset) % 129
        if val not in candidates:
            candidates.append(val)
    
    # Fill remaining if needed
    if len(candidates) < 12:
        for v in range(129):
            if len(candidates) >= 12:
                break
            if v not in candidates:
                candidates.append(v)
    
    return sorted(candidates[:12])

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    random.seed(42)
    
    # Start with a known-good initial set
    best_set = list(range(12))
    best_size = _max_sidon_size(best_set)
    
    # Strategy 1: Modular constructions
    for mod in range(2, 20):
        if time.time() > deadline:
            break
        for offset in range(0, min(mod, 10)):
            if time.time() > deadline:
                break
            cand = _construct_modular_set(mod, offset)
            if len(set(cand)) == 12 and max(cand) <= 128:
                ms = _max_sidon_size(cand)
                if ms < best_size:
                    best_size = ms
                    best_set = cand
                    if best_size <= 2:
                        return best_set
    
    # Strategy 2: Collision graph local search
    if best_size > 2:
        best_set, best_size, done = _local_search_collision_graph(
            best_set, deadline, best_size, best_set, max_iters=3000)
        if done:
            return best_set
    
    # Strategy 3: Targeted dense constructions
    # Sets where many elements are in arithmetic progressions with small differences
    dense_patterns = [
        # Pairs that sum to same value: (0,10),(1,9),(2,8),(3,7),(4,6),(5,5)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # More clustered
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        # Two clusters
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
        # Geometric-like
        [0, 1, 2, 4, 8, 16, 32, 64, 12, 24, 48, 96],
        # Mixed
        [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18],
        # Very dense small range
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14],
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]
    
    for cand in dense_patterns:
        if time.time() > deadline:
            break
        if len(set(cand)) == 12 and max(cand) <= 128:
            ms = _max_sidon_size(cand)
            if ms < best_size:
                best_size = ms
                best_set = cand[:]
                if best_size <= 2:
                    return best_set
    
    # Strategy 4: Random search with density-guided acceptance
    for attempt in range(30000):
        if time.time() > deadline:
            break
        
        # Generate random set biased toward small values
        max_val = random.choice([15, 20, 25, 30, 40, 50])
        candidates = sorted(random.sample(range(max_val + 1), 12))
        
        # Quick filter: check if it has a 3-Sidon
        if _has_sidon3(candidates):
            continue
        
        ms = _max_sidon_size(candidates)
        if ms < best_size:
            best_size = ms
            best_set = candidates
            if best_size <= 2:
                return best_set
    
    # Strategy 5: More local search from current best
    if best_size > 2:
        best_set, best_size, done = _local_search_collision_graph(
            best_set, deadline, best_size, best_set, max_iters=5000)
        if done:
            return best_set
    
    # Strategy 6: Try to build sets with no 3-Sidon at all
    # A 3-set {a,b,c} is NOT Sidon iff b=(a+c)/2 or c=2a-b or b=2a-c (with appropriate ordering)
    # So we want every triple to have a "midpoint" relationship
    # This is very restrictive. Let's try structured sets.
    for a in range(0, 30):
        if time.time() > deadline:
            break
        for d in range(1, 10):
            if time.time() > deadline:
                break
            # Arithmetic progression: every triple has midpoint
            cand = [a + i*d for i in range(12)]
            if max(cand) > 128:
                continue
            if len(set(cand)) == 12:
                ms = _max_sidon_size(cand)
                if ms < best_size:
                    best_size = ms
                    best_set = cand
                    if best_size <= 2:
                        return best_set
    
    # Strategy 7: Final local search polish
    if best_size > 2:
        best_set, best_size, done = _local_search_collision_graph(
            best_set, deadline, best_size, best_set, max_iters=8000)
        if done:
            return best_set
    
    return best_set

# EVOLVE-BLOCK-END
