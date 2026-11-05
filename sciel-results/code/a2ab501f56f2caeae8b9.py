import time
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Precompute adjacency
_N = P['n']
_ADJ = [set() for _ in range(_N)]
for u, v in P['edges']:
    _ADJ[u].add(v)
    _ADJ[v].add(u)

def _greedy_path(start, adj, rng=None):
    """Greedy path from start, choosing neighbor with fewest unvisited options."""
    path = [start]
    seen = {start}
    while True:
        candidates = adj[path[-1]] - seen
        if not candidates:
            break
        if rng:
            # Randomized greedy
            # Pick from candidates with smallest degree among unvisited
            min_deg = min(len(adj[w] - seen) for w in candidates)
            best = [w for w in candidates if len(adj[w] - seen) == min_deg]
            v = rng.choice(best)
        else:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def _swap_improve(path, adj, max_iters=200):
    """Try to extend path by finding a vertex that can be inserted or by segment reversal."""
    path = list(path)
    n = len(path)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try inserting an unvisited vertex between two consecutive vertices
        in_path = set(path)
        unvisited = [v for v in range(_N) if v not in in_path]
        for i in range(n - 1):
            a, b = path[i], path[i+1]
            for v in unvisited:
                if v in adj[a] and v in adj[b]:
                    # Insert v between a and b
                    path.insert(i+1, v)
                    in_path.add(v)
                    n = len(path)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # Try extending endpoints
        # Extend left
        last_left = path[0]
        for v in adj[last_left]:
            if v not in in_path:
                path.insert(0, v)
                in_path.add(v)
                n = len(path)
                improved = True
                break
        if improved:
            continue
        # Extend right
        last_right = path[-1]
        for v in adj[last_right]:
            if v not in in_path:
                path.append(v)
                in_path.add(v)
                n = len(path)
                improved = True
                break
    return path

def _two_opt(path, adj):
    """Try 2-opt style improvements: reverse a segment to potentially allow extension."""
    path = list(path)
    n = len(path)
    in_path = set(path)
    improved = True
    while improved:
        improved = False
        # Try reversing segments to change endpoints
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Consider reversing path[i..j]
                # New endpoints would be path[i-1] (or start) and path[j+1] (or end)
                # Actually 2-opt for path: we want to see if reversing helps extend
                # Let's just try: reverse a prefix or suffix
                pass
        
        # Simpler: try to find two non-adjacent edges (a,b) and (c,d) in path
        # such that a-d and b-c are edges, and reversing the segment between b and c
        # gives a valid path with same length but different endpoints
        for i in range(n - 1):
            a, b = path[i], path[i+1]
            for j in range(i + 2, n - 1):
                c, d = path[j], path[j+1]
                if a in adj[d] and b in adj[c]:
                    # Reverse segment path[i+1..j]
                    new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                    # Check validity (should be valid by construction)
                    # Try to extend
                    new_in = set(new_path)
                    ext = _swap_improve(new_path, adj, max_iters=20)
                    if len(ext) > len(path):
                        path = ext
                        n = len(path)
                        in_path = set(path)
                        improved = True
                        break
            if improved:
                break
    return path

def _dfs_longest_path(start, adj, time_limit=5.0):
    """DFS with backtracking to find long path from start."""
    best = []
    start_time = time.time()
    
    # Use iterative DFS with stack to avoid recursion limits
    # State: (current_node, path_so_far, visited_set)
    # This can be expensive, so we limit depth and use pruning
    
    def dfs(node, path, visited):
        nonlocal best, start_time
        if time.time() - start_time > time_limit:
            return
        if len(path) > len(best):
            best = list(path)
        
        candidates = [v for v in adj[node] if v not in visited]
        if not candidates:
            return
        
        # Order candidates by degree (fewest first) for better pruning
        candidates.sort(key=lambda w: (len(adj[w] - visited), w))
        
        for v in candidates:
            if time.time() - start_time > time_limit:
                return
            path.append(v)
            visited.add(v)
            dfs(v, path, visited)
            path.pop()
            visited.remove(v)
    
    dfs(start, [start], {start})
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_path = []
    best_len = 0
    
    # Phase 1: Greedy from all starts
    for s in range(_N):
        if time.time() > deadline - 30.0:
            break
        p = _greedy_path(s, _ADJ)
        if len(p) > best_len:
            best_len = len(p)
            best_path = list(p)
    
    # Phase 2: Improve best path with swap_improve and 2-opt
    if best_path:
        improved = _swap_improve(best_path, _ADJ, max_iters=500)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = list(improved)
        
        improved2 = _two_opt(best_path, _ADJ)
        if len(improved2) > best_len:
            best_len = len(improved2)
            best_path = list(improved2)
    
    # Phase 3: DFS from promising starts (high-degree vertices and endpoints of best path)
    if time.time() < deadline - 10.0:
        # Sort vertices by degree descending
        deg_order = sorted(range(_N), key=lambda v: len(_ADJ[v]), reverse=True)
        # Also include endpoints of best path
        dfs_starts = list(deg_order[:10])
        if best_path:
            if best_path[0] not in dfs_starts:
                dfs_starts.append(best_path[0])
            if best_path[-1] not in dfs_starts:
                dfs_starts.append(best_path[-1])
        
        for s in dfs_starts:
            if time.time() > deadline - 10.0:
                break
            dfs_limit = min(5.0, max(0.5, deadline - 10.0 - time.time()))
            if dfs_limit < 0.5:
                break
            p = _dfs_longest_path(s, _ADJ, time_limit=dfs_limit)
            if len(p) > best_len:
                best_len = len(p)
                best_path = list(p)
                # Improve the new best
                improved = _swap_improve(best_path, _ADJ, max_iters=200)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_path = list(improved)
    
    # Phase 4: More greedy with randomization
    if time.time() < deadline - 5.0:
        rng = random.Random(42)
        for trial in range(50):
            if time.time() > deadline - 5.0:
                break
            s = rng.randint(0, _N - 1)
            p = _greedy_path(s, _ADJ, rng=rng)
            if len(p) > best_len:
                best_len = len(p)
                best_path = list(p)
            # Also try improving
            improved = _swap_improve(p, _ADJ, max_iters=100)
            if len(improved) > best_len:
                best_len = len(improved)
                best_path = list(improved)
    
    # Phase 5: Final polish on best path
    if best_path and time.time() < deadline - 1.0:
        final = _swap_improve(best_path, _ADJ, max_iters=1000)
        if len(final) > best_len:
            best_len = len(final)
            best_path = list(final)
        final2 = _two_opt(best_path, _ADJ)
        if len(final2) > best_len:
            best_len = len(final2)
            best_path = list(final2)
    
    # Ensure we return a valid nonempty path
    if not best_path:
        # Fallback: return a single vertex or an edge
        if _ADJ[0]:
            best_path = [0, list(_ADJ[0])[0]]
        else:
            best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
