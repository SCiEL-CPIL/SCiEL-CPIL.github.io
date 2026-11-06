import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140
    results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 3:
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = _instance(*spec)
        paths = _solve_instance(edges, deadline)
        results.append(paths)
        
    return results

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _lower_bound(edges):
    if not edges:
        return 0
    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    odd_count = sum(1 for d in degree.values() if d % 2 == 1)
    return max(1, (odd_count + 1) // 2)

def _solve_instance(edges, deadline):
    if not edges:
        return []
    
    lb = _lower_bound(edges)
    
    best_paths = None
    best_count = float('inf')
    
    # Many fast restarts
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 1:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        # Greedy decomposition with randomized Warnsdorff
        paths = _greedy_decompose(edges, rng, deadline)
        
        if paths is None:
            continue
        
        # Optimal merge via maximum bipartite matching
        paths = _max_matching_merge(paths, deadline)
        
        count = len(paths)
        if count < best_count:
            best_count = count
            best_paths = [list(p) for p in paths]
            if count <= lb:
                break
    
    return best_paths if best_paths is not None else _greedy_decompose(edges, random.Random(0), deadline)

def _greedy_decompose(edges, rng, deadline):
    remaining = set(tuple(sorted(e)) for e in edges)
    if not remaining:
        return []
    
    adj = defaultdict(set)
    degree = defaultdict(int)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1
        
    paths = []
    
    while remaining:
        if time.time() > deadline:
            return None
        
        # Pick start vertex: prefer odd degree
        candidates = [v for v in adj if adj[v]]
        if not candidates:
            break
            
        odd_verts = [v for v in candidates if degree.get(v, 0) % 2 == 1]
        if odd_verts:
            start = rng.choice(odd_verts)
        else:
            # Weight by degree for diversity
            weights = [max(1, degree.get(v, 0)) for v in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            start = candidates[-1]
            for v, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    start = v
                    break
            
        # Find long path using Warnsdorff with top-k selection
        path = [start]
        visited = {start}
        current = start
        
        while True:
            neighbors = [n for n in adj[current] if n not in visited]
            if not neighbors:
                break
            
            # Warnsdorff: prefer neighbors with fewer unvisited neighbors
            def score(nb):
                return sum(1 for x in adj[nb] if x not in visited)
            
            if rng.random() < 0.85:
                neighbors.sort(key=score)
                top_k = min(3, len(neighbors))
                next_v = rng.choice(neighbors[:top_k])
            else:
                next_v = rng.choice(neighbors)
            
            path.append(next_v)
            visited.add(next_v)
            
            # Remove edge
            edge = tuple(sorted((current, next_v)))
            remaining.discard(edge)
            adj[current].discard(next_v)
            adj[next_v].discard(current)
            degree[current] = degree.get(current, 0) - 1
            degree[next_v] = degree.get(next_v, 0) - 1
            
            if not adj[current]:
                del adj[current]
            if not adj[next_v]:
                del adj[next_v]
            
            current = next_v
            
        paths.append(path)
        
    return paths

def _max_matching_merge(paths, deadline):
    """Merge paths using maximum bipartite matching on endpoint compatibility.
    
    We model this as: each path has two endpoints. Two paths can be merged if
    one endpoint of path A equals one endpoint of path B, and the two paths are
    vertex-disjoint (except at the shared endpoint). We want to find the maximum
    number of such merges.
    
    This is a maximum matching problem on a graph where nodes are paths and edges
    represent valid merges. We use a simple augmenting path algorithm.
    """
    if len(paths) <= 1:
        return paths
    
    n = len(paths)
    path_sets = [set(p) for p in paths]
    
    # Build compatibility: compat[i][j] = True if paths i and j can be merged
    # and also store the merge direction
    compat = [[False] * n for _ in range(n)]
    merge_info = [[None] * n for _ in range(n)]  # stores merged result
    
    for i in range(n):
        if time.time() > deadline:
            return paths
        for j in range(i + 1, n):
            pi = paths[i]
            pj = paths[j]
            
            # Try all 4 merge types
            merged = None
            # end of i to start of j
            if pi[-1] == pj[0]:
                set_i = set(pi[:-1])
                set_j = set(pj)
                if not set_i & set_j:
                    merged = pi + pj[1:]
            # end of i to end of j (reverse j)
            if merged is None and pi[-1] == pj[-1]:
                rev_pj = pj[::-1]
                set_i = set(pi[:-1])
                set_j = set(rev_pj)
                if not set_i & set_j:
                    merged = pi + rev_pj[1:]
            # start of i to start of j (reverse i)
            if merged is None and pi[0] == pj[0]:
                rev_pi = pi[::-1]
                set_i = set(rev_pi[:-1])
                set_j = set(pj)
                if not set_i & set_j:
                    merged = rev_pi + pj[1:]
            # start of i to end of j (reverse j, then i)
            if merged is None and pi[0] == pj[-1]:
                rev_pj = pj[::-1]
                set_i = set(pi)
                set_j = set(rev_pj[:-1])
                if not set_i & set_j:
                    merged = rev_pj + pi[1:]
            
            if merged is not None:
                compat[i][j] = True
                compat[j][i] = True
                merge_info[i][j] = merged
                merge_info[j][i] = merged
    
    # Find maximum matching using augmenting paths
    # match[i] = j means path i is matched with path j
    match = [-1] * n
    
    def augment(u, visited):
        for v in range(n):
            if not compat[u][v] or visited[v]:
                continue
            if match[v] == -1 or augment(match[v], visited):
                match[u] = v
                match[v] = u
                return True
        return False
    
    for u in range(n):
        if time.time() > deadline:
            break
        if match[u] == -1:
            visited = [False] * n
            augment(u, visited)
    
    # Now apply the matches to actually merge paths
    # Collect matched pairs
    matched = set()
    merges = []
    for i in range(n):
        if i < match[i]:  # avoid double counting
            merges.append((i, match[i]))
            matched.add(i)
            matched.add(match[i])
    
    # Build new path list
    new_paths = []
    for i, j in merges:
        new_paths.append(list(merge_info[i][j]))
    for i in range(n):
        if i not in matched:
            new_paths.append(list(paths[i]))
    
    return new_paths

def _is_valid_path(path, edges):
    """Check if path is simple and uses only valid edges."""
    if len(path) < 2:
        return len(path) == 1
    
    # Check simplicity
    if len(set(path)) != len(path):
        return False
    
    # Check edges
    edge_set = set(tuple(sorted(e)) for e in edges)
    for i in range(len(path) - 1):
        e = tuple(sorted((path[i], path[i + 1])))
        if e not in edge_set:
            return False
    
    return True
