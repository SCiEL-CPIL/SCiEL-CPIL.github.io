"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']

# Build adjacency sets
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

# Also build a list of neighbors sorted by degree (ascending) for greedy preference
neighbors = [sorted(adj[i], key=lambda x: (len(adj[x]), x)) for i in range(N)]

def greedy_path(start, visited_mask=None):
    """Greedy path from start, preferring low-degree unvisited neighbors."""
    if visited_mask is None:
        visited_mask = 0
    path = [start]
    visited_mask |= (1 << start)
    while True:
        cur = path[-1]
        candidates = []
        for nb in neighbors[cur]:
            if not (visited_mask >> nb) & 1:
                candidates.append(nb)
        if not candidates:
            break
        # Pick the candidate with fewest unvisited neighbors (Warnsdorff-like)
        def score(v):
            cnt = 0
            for w in adj[v]:
                if not (visited_mask >> w) & 1:
                    cnt += 1
            return (cnt, w)
        v = min(candidates, key=score)
        path.append(v)
        visited_mask |= (1 << v)
    return path, visited_mask

def two_opt_improve(path):
    """Try 2-opt style reversals to extend the path.
    A path is a sequence v0..vk. We can try to reverse a subpath and see if
    the new endpoints connect to extend. Actually for simple paths, 2-opt
    doesn't directly apply the same way as TSP. Instead, we try:
    - For each pair (i, j) with j > i, check if reversing path[i..j] and
      reconnecting gives a valid longer path or same-length path that allows
      further extension.
    
    Simpler approach: try to insert unvisited vertices at the ends.
    """
    n = len(path)
    if n < 2:
        return path
    
    visited = set(path)
    improved = True
    while improved:
        improved = False
        # Try extending at the start
        for nb in neighbors[path[0]]:
            if nb not in visited:
                path.insert(0, nb)
                visited.add(nb)
                improved = True
                break
        # Try extending at the end
        if not improved:
            for nb in neighbors[path[-1]]:
                if nb not in visited:
                    path.append(nb)
                    visited.add(nb)
                    improved = True
                    break
        # Try 2-opt: reverse a segment to create new endpoints that can be extended
        if not improved:
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Check if reversing path[i..j] is valid
                    # New connections: path[i-1] - path[j] and path[i] - path[j+1] (if j < n-1)
                    if path[i-1] == path[j] or (j < n - 1 and path[i] == path[j+1]):
                        continue
                    if path[i-1] not in adj[path[j]]:
                        continue
                    if j < n - 1 and path[i] not in adj[path[j+1]]:
                        continue
                    # Perform reversal
                    path[i:j+1] = path[i:j+1][::-1]
                    improved = True
                    break
                if improved:
                    break
    return path

def local_search(path):
    """Iteratively try to extend and rearrange to find a longer path."""
    best = list(path)
    visited = set(best)
    
    # First, greedy extend both ends
    changed = True
    while changed:
        changed = False
        for nb in neighbors[best[0]]:
            if nb not in visited:
                best.insert(0, nb)
                visited.add(nb)
                changed = True
                break
        if not changed:
            for nb in neighbors[best[-1]]:
                if nb not in visited:
                    best.append(nb)
                    visited.add(nb)
                    changed = True
                    break
    
    # Try 2-opt reversals to enable further extension
    n = len(best)
    for _ in range(50):  # limited iterations
        found = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if best[i-1] == best[j] or (j < n - 1 and best[i] == best[j+1]):
                    continue
                if best[i-1] not in adj[best[j]]:
                    continue
                if j < n - 1 and best[i] not in adj[best[j+1]]:
                    continue
                # Try reversal
                old = list(best)
                best[i:j+1] = best[i:j+1][::-1]
                visited = set(best)
                # Try to extend
                ext = 0
                for nb in neighbors[best[0]]:
                    if nb not in visited:
                        best.insert(0, nb)
                        visited.add(nb)
                        ext += 1
                        break
                for nb in neighbors[best[-1]]:
                    if nb not in visited:
                        best.append(nb)
                        visited.add(nb)
                        ext += 1
                        break
                if ext > 0 or len(best) > len(old):
                    found = True
                    break
                else:
                    best = old
            if found:
                break
        if not found:
            break
    
    return best

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    best_path = []
    
    # Phase 1: Greedy from every start vertex
    for start in range(N):
        if time.time() > deadline:
            break
        path, _ = greedy_path(start)
        if len(path) > len(best_path):
            best_path = list(path)
    
    # Phase 2: Local search (2-opt + extension) on the best greedy path
    if best_path:
        improved = local_search(best_path)
        if len(improved) > len(best_path):
            best_path = improved
    
    # Phase 3: Try multiple random restarts with different greedy strategies
    import random
    rng = random.Random(42)
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Random start
        start = rng.randint(0, N - 1)
        
        # Greedy with random tie-breaking
        path = [start]
        visited = {start}
        while True:
            cur = path[-1]
            candidates = [nb for nb in adj[cur] if nb not in visited]
            if not candidates:
                break
            # Random choice among candidates, slightly biased toward low degree
            weights = [1.0 / (len(adj[v]) + 1) for v in candidates]
            v = rng.choices(candidates, weights=weights)[0]
            path.append(v)
            visited.add(v)
        
        # Local search
        improved = local_search(path)
        if len(improved) > len(best_path):
            best_path = improved
    
    # Phase 4: Try to extend best_path further with targeted search
    if best_path:
        visited = set(best_path)
        # Try each unvisited vertex to see if it can be inserted
        for v in range(N):
            if v in visited:
                continue
            if time.time() > deadline:
                break
            # Check if v can be inserted between two consecutive vertices
            for i in range(len(best_path) - 1):
                a, b = best_path[i], best_path[i+1]
                if v in adj[a] and v in adj[b]:
                    new_path = best_path[:i+1] + [v] + best_path[i+1:]
                    if len(new_path) > len(best_path):
                        best_path = new_path
                        visited.add(v)
                        break
    
    # Ensure we return a valid path
    if not best_path:
        return [0]
    
    # Final validation
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in adj[best_path[i]]:
            # Truncate to valid prefix
            return best_path[:i+1]
    
    return best_path
# EVOLVE-BLOCK-END
