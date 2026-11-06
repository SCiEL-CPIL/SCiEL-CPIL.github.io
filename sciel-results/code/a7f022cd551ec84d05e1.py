"""Find long simple path in the pinned G(100, 1.8/100) graph."""
import time
import random
import sys

# Pin the graph exactly as in the seed program
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']

def _build_adj():
    adj = [set() for _ in range(N)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_path(adj, start, seen=None, strategy='min_deg'):
    """Greedy path from start. strategy: 'min_deg', 'max_deg', 'random'."""
    if seen is None:
        seen = set()
    path = [start]
    seen.add(start)
    while True:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        if strategy == 'min_deg':
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'max_deg':
            v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
        else:
            v = min(candidates, key=lambda w: (random.random(), w))
        path.append(v)
        seen.add(v)
    return path

def _extend_both_ends(adj, path):
    """Try to extend the path from both ends."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        # Try extending the start
        cur = path[0]
        candidates = adj[cur] - seen
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Try extending the end
        cur = path[-1]
        candidates = adj[cur] - seen
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def _backtrack_dfs(adj, path, seen, deadline, best_path, best_len, max_depth=50):
    """Limited-depth DFS backtracking to try to extend the current path."""
    cur = path[-1]
    candidates = sorted(adj[cur] - seen, key=lambda w: (len(adj[w] - seen), w))
    if len(path) >= max_depth or len(path) >= best_len + max_depth:
        return
    for v in candidates:
        if time.time() > deadline:
            return
        path.append(v)
        seen.add(v)
        if len(path) > best_len:
            best_path[:] = path[:]
            best_len[0] = len(path)
        # Recurse with limited depth
        _backtrack_dfs(adj, path, seen, deadline, best_path, best_len, max_depth)
        path.pop()
        seen.discard(v)

def _random_restart_greedy(adj, rng, deadline, best_path, best_len):
    """Randomized greedy with different strategies."""
    while time.time() < deadline:
        start = rng.randint(0, N - 1)
        strategy = rng.choice(['min_deg', 'max_deg', 'random'])
        path = _greedy_path(adj, start, strategy=strategy)
        path = _extend_both_ends(adj, path)
        if len(path) > best_len[0]:
            best_path[:] = path[:]
            best_len[0] = len(path)

def _two_end_growth(adj, rng, deadline, best_path, best_len):
    """Start from random edge and grow both ends."""
    edges = list(P['edges'])
    while time.time() < deadline:
        u, v = rng.choice(edges)
        path = [u, v]
        seen = {u, v}
        # Grow from both ends
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Extend start
            cur = path[0]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v)
                seen.add(v)
                changed = True
            # Extend end
            cur = path[-1]
            candidates = adj[cur] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
                seen.add(v)
                changed = True
        if len(path) > best_len[0]:
            best_path[:] = path[:]
            best_len[0] = len(path)

def _local_search(adj, path, deadline, best_path, best_len):
    """Try to improve path by local modifications: swap endpoints, try different extensions."""
    if len(path) < best_len[0]:
        return
    seen = set(path)
    # Try replacing the last vertex with a different neighbor
    if len(path) >= 2:
        cur = path[-2]
        candidates = adj[cur] - (seen - {path[-1]})
        for v in candidates:
            if v == path[-1]:
                continue
            new_path = path[:-1] + [v]
            # Try to extend from v
            new_seen = set(new_path)
            while True:
                c = new_path[-1]
                cands = adj[c] - new_seen
                if not cands:
                    break
                w = min(cands, key=lambda x: (len(adj[x] - new_seen), x))
                new_path.append(w)
                new_seen.add(w)
            if len(new_path) > best_len[0]:
                best_path[:] = new_path[:]
                best_len[0] = len(new_path)
            if time.time() > deadline:
                return

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    adj = _build_adj()
    rng = random.Random(42)
    
    best_path = []
    best_len = [0]
    
    # Phase 1: Deterministic greedy from all starts with multiple strategies
    for start in range(N):
        for strategy in ['min_deg', 'max_deg']:
            if time.time() > deadline:
                break
            path = _greedy_path(adj, start, strategy=strategy)
            path = _extend_both_ends(adj, path)
            if len(path) > best_len[0]:
                best_path = path
                best_len[0] = len(path)
    
    # Phase 2: Randomized restarts
    _random_restart_greedy(adj, rng, deadline, best_path, best_len)
    
    # Phase 3: Two-end growth from random edges
    _two_end_growth(adj, rng, deadline, best_path, best_len)
    
    # Phase 4: Local search improvements on the best path found
    if time.time() < deadline:
        _local_search(adj, best_path, deadline, best_path, best_len)
    
    # Phase 5: More randomized exploration with different seeds
    rng2 = random.Random(123)
    _random_restart_greedy(adj, rng2, deadline, best_path, best_len)
    
    # Final: ensure we have a valid nonempty path
    if not best_path:
        # Fallback: just return a single vertex
        best_path = [0]
    
    return best_path
