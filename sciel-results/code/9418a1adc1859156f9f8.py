"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 55  # 55s safety margin
    
    best_path = []
    
    def greedy_extend(path, seen, strategy, rng):
        """Extend path greedily using given strategy."""
        while True:
            cur = path[-1]
            candidates = adj[cur] - seen
            if not candidates:
                break
            if strategy == 'min_deg':
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            elif strategy == 'max_deg':
                v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
            else:  # random
                v = rng.choice(sorted(candidates))
            path.append(v)
            seen.add(v)
        return path
    
    def try_path(path):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = path[:]
        return len(path)
    
    # Strategy 1: Greedy with min-degree, all starting vertices
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        path = greedy_extend(path, seen, 'min_deg', random.Random(0))
        try_path(path)
    
    # Strategy 2: Greedy with max-degree, all starting vertices
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        path = greedy_extend(path, seen, 'max_deg', random.Random(0))
        try_path(path)
    
    # Strategy 3: Randomized greedy with many restarts
    rng = random.Random(42)
    for restart in range(2000):
        if time.time() > deadline:
            break
        start = rng.randint(0, n - 1)
        path = [start]
        seen = {start}
        path = greedy_extend(path, seen, 'random', rng)
        try_path(path)
    
    # Strategy 4: Randomized greedy with min-degree bias
    rng = random.Random(123)
    for restart in range(2000):
        if time.time() > deadline:
            break
        start = rng.randint(0, n - 1)
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            candidates = adj[cur] - seen
            if not candidates:
                break
            # Weighted random: prefer lower degree
            weights = [1.0 / (len(adj[w] - seen) + 1) for w in candidates]
            total = sum(weights)
            r = rng.random() * total
            cum = 0
            v = list(candidates)[0]
            for w, wt in zip(candidates, weights):
                cum += wt
                if r <= cum:
                    v = w
                    break
            path.append(v)
            seen.add(v)
        try_path(path)
    
    # Strategy 5: Local search - try to extend best path from both ends
    if best_path and time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            path = best_path[:]
            seen = set(path)
            # Try extending from end
            cur = path[-1]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
                improved = True
            # Try extending from front
            cur = path[0]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v)
                seen.add(v)
                improved = True
            if len(path) > len(best_path):
                best_path = path
                improved = True
    
    # Strategy 6: More aggressive randomized restarts with mixed strategies
    rng = random.Random(999)
    for restart in range(5000):
        if time.time() > deadline:
            break
        start = rng.randint(0, n - 1)
        path = [start]
        seen = {start}
        strat = rng.choice(['min_deg', 'max_deg', 'random'])
        path = greedy_extend(path, seen, strat, rng)
        try_path(path)
    
    # Strategy 7: Try to find longer paths by starting from high-degree vertices first
    degs = sorted(range(n), key=lambda x: -len(adj[x]))
    for start in degs:
        if time.time() > deadline:
            break
        for strat in ['min_deg', 'max_deg']:
            path = [start]
            seen = {start}
            path = greedy_extend(path, seen, strat, random.Random(0))
            try_path(path)
    
    # Strategy 8: Random walk with backtracking (limited)
    rng = random.Random(777)
    for restart in range(3000):
        if time.time() > deadline:
            break
        start = rng.randint(0, n - 1)
        path = [start]
        seen = {start}
        # Random walk
        while True:
            cur = path[-1]
            candidates = adj[cur] - seen
            if not candidates:
                break
            v = rng.choice(sorted(candidates))
            path.append(v)
            seen.add(v)
        try_path(path)
    
    # Strategy 9: Try all pairs as potential path extensions
    if time.time() < deadline:
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    path = [u, v]
                    seen = {u, v}
                    path = greedy_extend(path, seen, 'min_deg', random.Random(0))
                    try_path(path)
                    path2 = [v, u]
                    seen2 = {u, v}
                    path2 = greedy_extend(path2, seen2, 'min_deg', random.Random(0))
                    try_path(path2)
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
    
    # Final local search on best path
    if best_path and time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            path = best_path[:]
            seen = set(path)
            changed = False
            # Try inserting unused vertices
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                for w in range(n):
                    if w not in seen and w in adj[a] and w in adj[b]:
                        path.insert(i+1, w)
                        seen.add(w)
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                # Try extending ends
                cur = path[-1]
                cands = adj[cur] - seen
                if cands:
                    v = min(cands, key=lambda w: (len(adj[w] - seen), w))
                    path.append(v)
                    changed = True
                if not changed:
                    cur = path[0]
                    cands = adj[cur] - seen
                    if cands:
                        v = min(cands, key=lambda w: (len(adj[w] - seen), w))
                        path.insert(0, v)
                        changed = True
            if len(path) > len(best_path):
                best_path = path
            else:
                break
    
    if not best_path:
        return [0]
    return best_path
# EVOLVE-BLOCK-END
