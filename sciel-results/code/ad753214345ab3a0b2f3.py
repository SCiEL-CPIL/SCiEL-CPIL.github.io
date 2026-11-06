"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_path(adj, start, rng, strategy='min_remaining'):
    """Greedy path construction with different strategies."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        if strategy == 'min_remaining':
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'max_remaining':
            v = max(candidates, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'random':
            v = rng.choice(sorted(candidates))
        else:
            v = min(candidates)
        path.append(v)
        seen.add(v)
    return path

def _try_extend_path(path, adj, rng, max_extensions=50):
    """Try to extend a path from either end."""
    if not path:
        return path
    path = list(path)
    seen = set(path)
    extended = True
    count = 0
    while extended and count < max_extensions:
        extended = False
        count += 1
        # Try extending from the end
        if path:
            cur = path[-1]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
                extended = True
        # Try extending from the beginning
        if path:
            cur = path[0]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v)
                seen.add(v)
                extended = True
    return path

def _two_opt_improve(path, adj, rng, max_iter=200):
    """Improve path length using 2-opt style swaps."""
    if len(path) < 3:
        return path
    path = list(path)
    seen = set(path)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        # Try to find a vertex not in path that can be inserted
        # For each edge (path[i], path[i+1]), check if there's a vertex v
        # such that path[i]-v and v-path[i+1] are edges
        for i in range(len(path) - 1):
            u, w = path[i], path[i + 1]
            # Find common neighbors of u and w that are not in seen
            common = (adj[u] & adj[w]) - seen
            if common:
                # Pick one
                v = min(common)
                # Insert v between u and w
                path = path[:i+1] + [v] + path[i+1:]
                seen.add(v)
                improved = True
                break
        # Also try to extend ends
        if not improved and path:
            cur = path[-1]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
                improved = True
        if not improved and path:
            cur = path[0]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v)
                seen.add(v)
                improved = True
    return path

def _random_walk_path(adj, rng, min_len=5):
    """Generate a random path."""
    start = rng.randint(0, P['n'] - 1)
    path = [start]
    seen = {start}
    while len(path) < min_len:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        v = rng.choice(sorted(candidates))
        path.append(v)
        seen.add(v)
    return path

def _merge_paths(path1, path2, adj):
    """Try to merge two paths into one longer path."""
    if not path1 or not path2:
        return max(path1, path2, key=len) if path1 or path2 else []
    p1 = set(path1)
    p2 = set(path2)
    if p1 & p2:
        return max(path1, path2, key=len)
    
    # Try to connect end of path1 to start of path2
    # or end of path1 to end of path2, etc.
    best = max(path1, path2, key=len)
    
    # Try all four combinations
    combos = [
        (path1, path2),
        (path1, path2[::-1]),
        (path1[::-1], path2),
        (path1[::-1], path2[::-1]),
    ]
    for a, b in combos:
        if not a or not b:
            continue
        # Check if we can connect a[-1] to b[0]
        if a[-1] in adj[b[0]]:
            merged = a + b
            if len(merged) > len(best):
                best = merged
        # Check if we can connect b[-1] to a[0]
        if b[-1] in adj[a[0]]:
            merged = b + a
            if len(merged) > len(best):
                best = merged
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    n = P['n']
    adj = _build_adj()
    rng = random.Random(748)
    
    best_path = []
    
    # Strategy 1: Greedy from every start vertex with multiple strategies
    for start in range(n):
        for strat in ['min_remaining', 'max_remaining']:
            path = _greedy_path(adj, start, rng, strategy=strat)
            if len(path) > len(best_path):
                best_path = path
    
    # Strategy 2: Improve best path with 2-opt
    if len(best_path) > 0:
        improved = _two_opt_improve(best_path, adj, rng, max_iter=500)
        if len(improved) > len(best_path):
            best_path = improved
    
    # Strategy 3: Random restarts
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        # Generate a random starting path
        path = _random_walk_path(adj, rng, min_len=3)
        # Extend it greedily
        path = _try_extend_path(path, adj, rng, max_extensions=100)
        # Improve with 2-opt
        path = _two_opt_improve(path, adj, rng, max_iter=100)
        if len(path) > len(best_path):
            best_path = path
        
        # Also try greedy from random starts
        for _ in range(5):
            s = rng.randint(0, n - 1)
            gp = _greedy_path(adj, s, rng, strategy='min_remaining')
            gp = _two_opt_improve(gp, adj, rng, max_iter=50)
            if len(gp) > len(best_path):
                best_path = gp
    
    # Strategy 4: Try to merge multiple good paths
    # Generate several candidate paths and try merging
    candidates = []
    for s in range(0, n, 2):
        if time.time() > deadline:
            break
        for strat in ['min_remaining', 'max_remaining']:
            p = _greedy_path(adj, s, rng, strategy=strat)
            p = _two_opt_improve(p, adj, rng, max_iter=30)
            if len(p) > len(best_path):
                best_path = p
            if len(p) >= 3:
                candidates.append(p)
    
    # Try merging pairs of candidates
    if len(candidates) >= 2 and time.time() < deadline:
        # Sort by length descending
        candidates.sort(key=len, reverse=True)
        for i in range(min(10, len(candidates))):
            if time.time() > deadline:
                break
            for j in range(i + 1, min(10, len(candidates))):
                merged = _merge_paths(candidates[i], candidates[j], adj)
                if len(merged) > len(best_path):
                    best_path = merged
                # Also try 2-opt on the merged path
                if len(merged) > len(best_path):
                    improved = _two_opt_improve(merged, adj, rng, max_iter=50)
                    if len(improved) > len(best_path):
                        best_path = improved
    
    # Final polish: try to extend the best path further
    if len(best_path) > 0 and time.time() < deadline:
        final = _try_extend_path(best_path, adj, rng, max_extensions=200)
        final = _two_opt_improve(final, adj, rng, max_iter=200)
        if len(final) > len(best_path):
            best_path = final
    
    # Ensure we return a valid nonempty path
    if not best_path:
        # Fallback: find any edge
        for u, v in P['edges']:
            best_path = [u, v]
            break
        if not best_path:
            best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
