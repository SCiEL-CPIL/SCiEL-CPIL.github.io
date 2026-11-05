"""Search-based program for long paths in a sparse graph."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGE_LIST = P['edges']


def _build_adj():
    adj = [set() for _ in range(N)]
    for u, v in EDGE_LIST:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _greedy_path(adj, start, rng):
    """Greedy path extension from start with random tie-breaking."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        neighbors = adj[cur] - seen
        if not neighbors:
            break
        # Choose neighbor with fewest unvisited neighbors (Warnsdorff-like)
        # with some randomness
        scored = []
        for v in neighbors:
            deg = len(adj[v] - seen)
            scored.append((deg, rng.random(), v))
        scored.sort()
        # Pick from top 3 with some probability, else pick best
        if len(scored) >= 3 and rng.random() < 0.3:
            v = scored[rng.randint(0, 2)][2]
        else:
            v = scored[0][2]
        path.append(v)
        seen.add(v)
    return path


def _greedy_from_both_ends(adj, start, rng):
    """Try to extend from both ends simultaneously."""
    path = [start]
    seen = {start}
    # Alternate between extending left and right
    left_ok = True
    right_ok = True
    while left_ok or right_ok:
        if left_ok and (not right_ok or rng.random() < 0.5):
            cur = path[0]
            neighbors = adj[cur] - seen
            if neighbors:
                scored = sorted([(len(adj[v] - seen), rng.random(), v) for v in neighbors])
                v = scored[0][2]
                path.insert(0, v)
                seen.add(v)
            else:
                left_ok = False
        if right_ok:
            cur = path[-1]
            neighbors = adj[cur] - seen
            if neighbors:
                scored = sorted([(len(adj[v] - seen), rng.random(), v) for v in neighbors])
                v = scored[0][2]
                path.append(v)
                seen.add(v)
            else:
                right_ok = False
    return path


def _randomized_walk(adj, start, rng, max_restarts=10):
    """Randomized greedy with restarts."""
    best = []
    for _ in range(max_restarts):
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            neighbors = adj[cur] - seen
            if not neighbors:
                break
            # Convert to list for consistent indexing and weight calculation
            neighbor_list = list(neighbors)
            # Random choice with bias toward lower degree
            weights = [1.0 / (len(adj[v] - seen) + 1) for v in neighbor_list]
            total = sum(weights)
            r = rng.random() * total
            cum = 0
            chosen = neighbor_list[0]
            for v, w in zip(neighbor_list, weights):
                cum += w
                if r <= cum:
                    chosen = v
                    break
            path.append(chosen)
            seen.add(chosen)
        if len(path) > len(best):
            best = path
    return best


def _try_extend_path(adj, path, seen):
    """Try to extend the path from both ends."""
    path = list(path)
    seen = set(seen)
    improved = True
    while improved:
        improved = False
        # Try extending right end
        cur = path[-1]
        neighbors = adj[cur] - seen
        if neighbors:
            # Pick the one with fewest remaining neighbors
            v = min(neighbors, key=lambda x: (len(adj[x] - seen), x))
            path.append(v)
            seen.add(v)
            improved = True
        # Try extending left end
        if not improved:
            cur = path[0]
            neighbors = adj[cur] - seen
            if neighbors:
                v = min(neighbors, key=lambda x: (len(adj[x] - seen), x))
                path.insert(0, v)
                seen.add(v)
                improved = True
    return path


def _two_opt_improve(adj, path):
    """Attempt 2-opt style improvements: try reversing segments to extend."""
    path = list(path)
    n = len(path)
    if n < 3:
        return path
    best = path
    best_len = n
    
    # Try to find extensions by checking if endpoints can connect to other parts
    # This is a simplified approach: try swapping adjacent pairs
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 1):
            # Try moving path[i] to the end
            if i < len(path) - 1:
                # Check if path[i] connects to path[-1]
                if path[-1] in adj[path[i]]:
                    # And path[i-1] connects to path[i+1]
                    if path[i + 1] in adj[path[i - 1]]:
                        new_path = path[:i] + path[i+1:] + [path[i]]
                        # Verify it's valid
                        valid = True
                        for j in range(len(new_path) - 1):
                            if new_path[j+1] not in adj[new_path[j]]:
                                valid = False
                                break
                        if valid and len(new_path) == len(path):
                            # Try to extend
                            new_seen = set(new_path)
                            new_path = _try_extend_path(adj, new_path, new_seen)
                            if len(new_path) > len(best):
                                best = new_path
                                best_len = len(new_path)
                                improved = True
                                break
    return best


def _backtrack_extend(adj, path, seen, deadline):
    """Backtracking search to extend path from the end."""
    path = list(path)
    seen = set(seen)
    
    def dfs(cur, depth):
        if time.time() > deadline:
            return
        if depth > 15:  # Limit recursion depth
            return
        neighbors = adj[cur] - seen
        if not neighbors:
            return
        # Sort by degree
        for v in sorted(neighbors, key=lambda x: (len(adj[x] - seen), x)):
            path.append(v)
            seen.add(v)
            dfs(v, depth + 1)
            path.pop()
            seen.discard(v)
            if time.time() > deadline:
                return
    
    dfs(path[-1], 0)
    return path


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    
    adj = _build_adj()
    rng = random.Random(748)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Diverse greedy constructions from all starts
    for start in range(N):
        if time.time() > deadline:
            break
        # Multiple strategies per start
        for strategy in range(5):
            if time.time() > deadline:
                break
            if strategy == 0:
                path = _greedy_path(adj, start, rng)
            elif strategy == 1:
                path = _greedy_from_both_ends(adj, start, rng)
            elif strategy == 2:
                path = _randomized_walk(adj, start, rng, max_restarts=3)
            else:
                # Random start variant
                path = _greedy_path(adj, start, rng)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
    
    # Phase 2: Local improvement on best paths
    if time.time() < deadline:
        # Try to extend best_path
        seen = set(best_path)
        improved = _try_extend_path(adj, best_path, seen)
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
    
    # Phase 3: More diverse searches
    if time.time() < deadline:
        # Try random starts with more aggressive randomization
        for _ in range(200):
            if time.time() > deadline:
                break
            start = rng.randint(0, N - 1)
            path = _randomized_walk(adj, start, rng, max_restarts=2)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
    
    # Phase 4: Backtracking on the best path's endpoints
    if time.time() < deadline - 5.0:
        seen = set(best_path)
        extended = _backtrack_extend(adj, best_path, seen, deadline - 2.0)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(extended)
    
    # Phase 5: Final extension attempts
    if time.time() < deadline:
        seen = set(best_path)
        final = _try_extend_path(adj, best_path, seen)
        if len(final) > best_len:
            best_path = final
            best_len = len(final)
    
    # Ensure we return a valid non-empty path
    if not best_path:
        # Fallback: just return a single vertex
        best_path = [0]
    
    # Final validation
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj[best_path[i]]:
            # Truncate to last valid point
            valid_len = i + 1
            best_path = best_path[:valid_len]
            break
    
    return best_path
# EVOLVE-BLOCK-END
