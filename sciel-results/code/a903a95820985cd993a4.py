"""Rewrite: strongest simple-path search within the greedy+refinement family."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
# Build adjacency
_adj = [set() for _ in range(N)]
for u, v in P['edges']:
    _adj[u].add(v)
    _adj[v].add(u)
# Precompute neighbor lists sorted by degree (ascending) for Warnsdorff
_deg = [len(_adj[i]) for i in range(N)]
_nbrs = [sorted(_adj[i], key=lambda x: (_deg[x], x)) for i in range(N)]

def _warnsdorff(start, seen=None):
    """Greedy Warnsdorff: always pick the neighbor with fewest unvisited neighbors."""
    if seen is None:
        seen = set()
    seen.add(start)
    path = [start]
    while True:
        cur = path[-1]
        cands = [v for v in _nbrs[cur] if v not in seen]
        if not cands:
            break
        # Warnsdorff: pick neighbor with minimum remaining unvisited degree
        best = min(cands, key=lambda v: (sum(1 for w in _nbrs[v] if w not in seen), v))
        path.append(best)
        seen.add(best)
    return path

def _greedy_start_degree(start, mode='min'):
    """Greedy by degree heuristic."""
    seen = {start}
    path = [start]
    while True:
        cur = path[-1]
        cands = [v for v in _nbrs[cur] if v not in seen]
        if not cands:
            break
        if mode == 'min':
            best = min(cands, key=lambda v: _deg[v])
        elif mode == 'max':
            best = max(cands, key=lambda v: _deg[v])
        else:  # random
            best = random.Random(hash((start, len(path))) ^ len(path)).choice(cands)
        path.append(best)
        seen.add(best)
    return path

def _extend(path, seen):
    """Try to extend the path from both ends using all possible greedy strategies."""
    # Try extending from the end
    improved = True
    while improved:
        improved = False
        # Extend from tail
        cur = path[-1]
        cands = [v for v in _nbrs[cur] if v not in seen]
        if cands:
            best = min(cands, key=lambda v: (sum(1 for w in _nbrs[v] if w not in seen), v))
            path.append(best)
            seen.add(best)
            improved = True
        # Extend from head (prepend)
        cur = path[0]
        cands = [v for v in _nbrs[cur] if v not in seen]
        if cands:
            best = min(cands, key=lambda v: (sum(1 for w in _nbrs[v] if w not in seen), v))
            path.insert(0, best)
            seen.add(best)
            improved = True
    return path

def _try_insert(path, seen):
    """Try to insert an unvisited vertex into the path at any valid position."""
    unvisited = [v for v in range(N) if v not in seen]
    for v in unvisited:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if b in _adj[v] and a in _adj[v]:
                new_path = path[:i+1] + [v] + path[i+1:]
                new_seen = seen | {v}
                return new_path, new_seen
    # Also try adding to ends
    for v in unvisited:
        if path[-1] in _adj[v]:
            return path + [v], seen | {v}
        if path[0] in _adj[v]:
            return [v] + path, seen | {v}
    return None

def _refine(path):
    """Local refinement: try insertions and re-extensions repeatedly."""
    seen = set(path)
    best_path = list(path)
    best_seen = set(seen)
    for _ in range(200):
        result = _try_insert(best_path, best_seen)
        if result is None:
            break
        new_path, new_seen = result
        if len(new_path) > len(best_path):
            best_path = new_path
            best_seen = new_seen
        else:
            break
    # Re-extend from both ends
    best_path = _extend(best_path, best_seen)
    return best_path

def _backtrack_extend(path, seen, depth_limit=4):
    """Limited backtracking to try alternative choices near the end."""
    if depth_limit <= 0:
        return path
    cur = path[-1]
    cands = [v for v in _nbrs[cur] if v not in seen]
    if not cands:
        return path
    best = list(path)
    for v in cands:
        trial = path + [v]
        trial_seen = seen | {v}
        # Greedy extend from here
        ext = _warnsdorff(v, trial_seen)
        if len(ext) > len(best):
            best = ext
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 75.0  # 75s safety margin within 90s limit
    
    best_path = []
    best_len = 0
    
    # Phase 1: Multi-start greedy with various heuristics
    starts = list(range(N))
    # Prioritize high-degree starts
    starts.sort(key=lambda s: -_deg[s])
    
    for start in starts:
        if time.time() > deadline - 5.0:
            break
        
        # Strategy 1: Warnsdorff
        p = _warnsdorff(start)
        if len(p) > best_len:
            best_path = list(p)
            best_len = len(p)
        
        # Strategy 2: Min-degree greedy
        p2 = _greedy_start_degree(start, 'min')
        if len(p2) > best_len:
            best_path = list(p2)
            best_len = len(p2)
        
        # Strategy 3: Max-degree greedy
        p3 = _greedy_start_degree(start, 'max')
        if len(p3) > best_len:
            best_path = list(p3)
            best_len = len(p3)
    
    # Phase 2: Refine the best path found
    if best_path:
        refined = _refine(best_path)
        if len(refined) > best_len:
            best_path = refined
            best_len = len(refined)
    
    # Phase 3: Try to improve via random restarts with different seeds
    rng = random.Random(12345)
    for iteration in range(500):
        if time.time() > deadline - 3.0:
            break
        
        start = rng.randint(0, N - 1)
        
        # Randomized Warnsdorff: shuffle candidates before picking
        seen = {start}
        path = [start]
        while True:
            cur = path[-1]
            cands = [v for v in _nbrs[cur] if v not in seen]
            if not cands:
                break
            # Shuffle and pick
            rng.shuffle(cands)
            # But still prefer low-degree (Warnsdorff-like)
            cands.sort(key=lambda v: (sum(1 for w in _nbrs[v] if w not in seen), v))
            # Pick from top few
            pick = cands[0] if len(cands) <= 2 else rng.choice(cands[:min(3, len(cands))])
            path.append(pick)
            seen.add(pick)
        
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        
        # Refine
        if len(path) >= best_len - 2:
            refined = _refine(path)
            if len(refined) > best_len:
                best_path = refined
                best_len = len(refined)
    
    # Phase 4: Final refinement of the best path
    if best_path:
        final = _refine(best_path)
        if len(final) > best_len:
            best_path = final
            best_len = len(final)
    
    # Phase 5: Try backtracking on the best path's end
    if best_path and time.time() < deadline - 2.0:
        seen = set(best_path)
        bt = _backtrack_extend(best_path, seen, depth_limit=3)
        if len(bt) > best_len:
            best_path = bt
            best_len = len(bt)
    
    # Ensure nonempty
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
