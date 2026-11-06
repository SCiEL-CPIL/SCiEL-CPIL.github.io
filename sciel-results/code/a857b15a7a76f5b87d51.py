"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    best = []
    best_len = 0
    
    def try_path(path):
        nonlocal best, best_len
        if len(path) > best_len:
            best = path[:]
            best_len = len(best)
    
    # Strategy 1: Greedy min-remaining (Warnsdorff-like)
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        while True:
            candidates = adj[path[-1]] - seen
            if not candidates:
                break
            # Prefer neighbors with fewest unvisited neighbors (to avoid trapping)
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
        try_path(path)
    
    # Strategy 2: Greedy max-remaining
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        while True:
            candidates = adj[path[-1]] - seen
            if not candidates:
                break
            v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
            path.append(v)
            seen.add(v)
        try_path(path)
    
    # Strategy 3: Randomized greedy with multiple trials
    rng = random.Random(42)
    for trial in range(800):
        if time.time() > deadline:
            break
        start = rng.randint(0, n - 1)
        path = [start]
        seen = {start}
        while True:
            candidates = list(adj[path[-1]] - seen)
            if not candidates:
                break
            # Mix of min-remaining and random
            if rng.random() < 0.7:
                candidates.sort(key=lambda w: (len(adj[w] - seen), rng.random()))
            else:
                rng.shuffle(candidates)
            v = candidates[0]
            path.append(v)
            seen.add(v)
        try_path(path)
    
    # Strategy 4: Backtracking with improved heuristics
    def backtrack_dfs(start, depth_limit=8):
        nonlocal best, best_len
        if time.time() > deadline:
            return
        path = [start]
        seen = {start}
        
        def _bt(path, seen, depth):
            nonlocal best, best_len
            if time.time() > deadline:
                return
            if len(path) > best_len:
                best = path[:]
                best_len = len(best)
            
            if depth >= depth_limit:
                return
            
            cur = path[-1]
            candidates = list(adj[cur] - seen)
            if not candidates:
                return
            
            # Sort by: fewest remaining neighbors first (Warnsdorff), then by degree
            candidates.sort(key=lambda w: (len(adj[w] - seen), len(adj[w]), w))
            
            # Try top candidates
            for v in candidates[:min(6, len(candidates))]:
                path.append(v)
                seen.add(v)
                _bt(path, seen, depth + 1)
                path.pop()
                seen.remove(v)
        
        _bt(path, seen, 0)
    
    for start in range(n):
        if time.time() > deadline:
            break
        backtrack_dfs(start, depth_limit=10)
    
    # Strategy 5: Path extension - try to extend the best path found
    if best:
        best_set = set(best)
        # Try extending from both ends
        for end_idx in [0, -1]:
            end_vertex = best[end_idx]
            candidates = list(adj[end_vertex] - best_set)
            if candidates:
                # Try each candidate
                for ext in candidates:
                    if end_idx == 0:
                        new_path = [ext] + best
                    else:
                        new_path = best + [ext]
                    # Try to extend further
                    extended = new_path[:]
                    ext_set = set(extended)
                    last = extended[-1] if end_idx == -1 else extended[0]
                    
                    # Continue greedy extension
                    if end_idx == -1:
                        while True:
                            cands = adj[extended[-1]] - ext_set
                            if not cands:
                                break
                            v = min(cands, key=lambda w: len(adj[w] - ext_set))
                            extended.append(v)
                            ext_set.add(v)
                    else:
                        while True:
                            cands = adj[extended[0]] - ext_set
                            if not cands:
                                break
                            v = min(cands, key=lambda w: len(adj[w] - ext_set))
                            extended.insert(0, v)
                            ext_set.add(v)
                    
                    if len(extended) > best_len:
                        best = extended
                        best_len = len(best)
    
    # Strategy 6: More aggressive backtracking with different ordering
    for start in range(n):
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        
        def _bt2(path, seen, depth):
            nonlocal best, best_len
            if time.time() > deadline:
                return
            if len(path) > best_len:
                best = path[:]
                best_len = len(best)
            
            if depth >= 12:
                return
            
            cur = path[-1]
            candidates = list(adj[cur] - seen)
            if not candidates:
                return
            
            # Different heuristic: prefer high-degree unvisited
            candidates.sort(key=lambda w: (-len(adj[w] - seen), -len(adj[w]), w))
            
            for v in candidates[:min(5, len(candidates))]:
                path.append(v)
                seen.add(v)
                _bt2(path, seen, depth + 1)
                path.pop()
                seen.remove(v)
        
        _bt2(path, seen, 0)
    
    # Strategy 7: Random restarts with backtracking
    rng2 = random.Random(123)
    for _ in range(100):
        if time.time() > deadline:
            break
        start = rng2.randint(0, n - 1)
        path = [start]
        seen = {start}
        
        def _bt3(path, seen, depth):
            nonlocal best, best_len
            if time.time() > deadline:
                return
            if len(path) > best_len:
                best = path[:]
                best_len = len(best)
            
            if depth >= 9:
                return
            
            cur = path[-1]
            candidates = list(adj[cur] - seen)
            if not candidates:
                return
            
            rng2.shuffle(candidates)
            # Take top 4 after random shuffle
            for v in candidates[:4]:
                path.append(v)
                seen.add(v)
                _bt3(path, seen, depth + 1)
                path.pop()
                seen.remove(v)
        
        _bt3(path, seen, 0)
    
    return best if best else [0]
# EVOLVE-BLOCK-END
