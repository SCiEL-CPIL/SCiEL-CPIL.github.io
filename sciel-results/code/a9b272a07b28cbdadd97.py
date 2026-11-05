"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

def _greedy_path(start, rng, strategy='min_remaining'):
    """Greedy path extension from start vertex."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = ADJ[cur] - seen
        if not candidates:
            break
        if strategy == 'min_remaining':
            # Choose neighbor with fewest unvisited neighbors (Warnsdorff-like)
            v = min(candidates, key=lambda w: (len(ADJ[w] - seen), w))
        elif strategy == 'max_remaining':
            v = max(candidates, key=lambda w: (len(ADJ[w] - seen), w))
        elif strategy == 'random':
            v = rng.choice(sorted(candidates))
        else:
            v = min(candidates)
        path.append(v)
        seen.add(v)
    return path

def _try_2opt(path, rng, max_iters=200):
    """Attempt to improve path length via 2-opt style reconnections.
    For a path (not a cycle), we try to splice: if we have path a...b...c...d
    and edge a-c exists and b-d doesn't conflict, we can potentially extend.
    Actually for simple paths, a better approach is: try to insert unvisited vertices.
    
    Better approach: try to extend the path at either end, or find a way to
    reconnect to include more vertices.
    """
    path = list(path)
    seen = set(path)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Try extending at the start
        if path:
            first = path[0]
            for nb in sorted(ADJ[first] - seen):
                new_path = [nb] + path
                if len(new_path) > len(path):
                    path = new_path
                    seen.add(nb)
                    improved = True
                    break
        
        # Try extending at the end
        if path:
            last = path[-1]
            for nb in sorted(ADJ[last] - seen):
                new_path = path + [nb]
                if len(new_path) > len(path):
                    path = new_path
                    seen.add(nb)
                    improved = True
                    break
        
        # Try to find a vertex that can be inserted or a reconnection
        # Check if there's an edge from path[i] to path[j] where we can split
        # and reconnect to include more vertices.
        # Simpler: try all pairs (i, j) with i < j-1 where edge path[i]-path[j] exists
        # This creates a cycle; then we can break the cycle at a different point.
        n = len(path)
        if n >= 4:
            for i in range(n - 2):
                for j in range(i + 2, n):
                    if path[j] in ADJ[path[i]]:
                        # We have edge path[i] - path[j]
                        # Current path: ... path[i] - path[i+1] - ... - path[j] - ...
                        # If we use edge path[i]-path[j], we form a cycle on path[i..j]
                        # The remaining parts are path[0..i-1] and path[j+1..n-1]
                        # We can try: path[j+1..n-1] + path[j..i] (reversed) + path[0..i-1]
                        # Or other combinations
                        # New path: path[j+1:] + path[i+1:j+1][::-1] + path[:i]
                        # Check this is valid
                        new_path = path[j+1:] + path[i+1:j+1][::-1] + path[:i]
                        # Verify it's a valid simple path
                        if len(new_path) == n and len(set(new_path)) == n:
                            # Check all consecutive pairs are edges
                            valid = True
                            for k in range(len(new_path) - 1):
                                if new_path[k+1] not in ADJ[new_path[k]]:
                                    valid = False
                                    break
                            if valid:
                                # Try to extend the new path
                                ns = set(new_path)
                                # Extend at start
                                if new_path:
                                    for nb in sorted(ADJ[new_path[0]] - ns):
                                        new_path = [nb] + new_path
                                        ns.add(nb)
                                        improved = True
                                        break
                                # Extend at end
                                if new_path:
                                    for nb in sorted(ADJ[new_path[-1]] - ns):
                                        new_path = new_path + [nb]
                                        ns.add(nb)
                                        improved = True
                                        break
                                if len(new_path) > len(path):
                                    path = new_path
                                    seen = set(path)
                                    break
                if improved:
                    break
    return path

def _dfs_path(start, limit, rng):
    """DFS-based path search with randomization, trying to find long paths."""
    best_path = [start]
    
    def dfs(path, seen):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = list(path)
        cur = path[-1]
        candidates = sorted(ADJ[cur] - seen, key=lambda x: rng.random())
        for v in candidates:
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.remove(v)
    
    dfs([start], {start})
    return best_path

def _simulated_annealing(rng, start_path, iters=5000):
    """Simple local search with random restarts and perturbations."""
    path = list(start_path)
    seen = set(path)
    best = list(path)
    best_len = len(path)
    
    for _ in range(iters):
        if len(path) == 0:
            break
        # Random perturbation: try to extend, remove an endpoint, or swap
        op = rng.random()
        if op < 0.4:
            # Try to extend at a random end
            if path:
                if rng.random() < 0.5:
                    end = path[0]
                    cands = sorted(ADJ[end] - seen)
                    if cands:
                        v = rng.choice(cands)
                        new_path = [v] + path
                        new_seen = seen | {v}
                        if len(new_path) > len(best):
                            best = new_path
                            best_len = len(new_path)
                        path = new_path
                        seen = new_seen
                else:
                    end = path[-1]
                    cands = sorted(ADJ[end] - seen)
                    if cands:
                        v = rng.choice(cands)
                        new_path = path + [v]
                        new_seen = seen | {v}
                        if len(new_path) > len(best):
                            best = new_path
                            best_len = len(new_path)
                        path = new_path
                        seen = new_seen
        elif op < 0.7:
            # Try to remove a random interior vertex and reconnect
            if len(path) >= 4:
                i = rng.randint(1, len(path) - 2)
                # Check if we can bypass path[i]
                prev = path[i-1]
                nxt = path[i+1]
                if nxt in ADJ[prev]:
                    new_path = path[:i] + path[i+1:]
                    new_seen = seen - {path[i]}
                    # Try to extend
                    if new_path:
                        for nb in sorted(ADJ[new_path[0]] - new_seen):
                            new_path = [nb] + new_path
                            new_seen.add(nb)
                            break
                    if new_path:
                        for nb in sorted(ADJ[new_path[-1]] - new_seen):
                            new_path = new_path + [nb]
                            new_seen.add(nb)
                            break
                    if len(new_path) >= len(path):
                        path = new_path
                        seen = new_seen
                        if len(path) > len(best):
                            best = list(path)
                            best_len = len(path)
        else:
            # Random restart with a small random path
            start = rng.randint(0, N - 1)
            path = _greedy_path(start, rng, 'random')
            seen = set(path)
            if len(path) > len(best):
                best = list(path)
                best_len = len(path)
    
    return best

def construct():
    t0 = time.time()
    deadline = 75.0  # safety margin before 90s
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Greedy from every start with multiple strategies
    for start in range(N):
        if time.time() > deadline:
            break
        for strategy in ['min_remaining', 'max_remaining', 'random']:
            path = _greedy_path(start, rng, strategy)
            if len(path) > best_len:
                best_path = list(path)
                best_len = len(path)
    
    # Phase 2: Local search improvement on best path
    if time.time() < deadline:
        improved = _try_2opt(best_path, rng, max_iters=100)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
    
    # Phase 3: Simulated annealing / random restarts
    if time.time() < deadline:
        sa_path = _simulated_annealing(rng, best_path, iters=3000)
        if len(sa_path) > best_len:
            best_path = sa_path
            best_len = len(sa_path)
    
    # Phase 4: More greedy attempts with different random seeds
    if time.time() < deadline:
        for trial in range(50):
            if time.time() > deadline:
                break
            start = rng.randint(0, N - 1)
            path = _greedy_path(start, rng, 'random')
            if len(path) > best_len:
                best_path = list(path)
                best_len = len(path)
    
    # Phase 5: Final local search
    if time.time() < deadline and best_path:
        improved = _try_2opt(best_path, rng, max_iters=200)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
    
    # Ensure validity
    if not best_path:
        # Fallback: find any edge
        for u, v in P['edges']:
            best_path = [u, v]
            break
        else:
            best_path = [0]
    
    # Validate
    seen = set()
    valid = True
    for i, v in enumerate(best_path):
        if v in seen:
            valid = False
            break
        seen.add(v)
        if i > 0 and v not in ADJ[best_path[i-1]]:
            valid = False
            break
    if not valid:
        # Rebuild a valid path
        best_path = _greedy_path(0, rng, 'min_remaining')
    
    return best_path
# EVOLVE-BLOCK-END
