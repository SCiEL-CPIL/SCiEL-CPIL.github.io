import time
import random
from collections import defaultdict, deque

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _greedy_longest_path(start, adj, remaining, n):
    """
    Find a long simple path starting from `start` using only edges in `remaining`.
    Uses a randomized greedy with backtracking on dead-ends to find longer paths.
    Returns (path, used_edges) where path is list of vertices, used_edges is set of edges used.
    """
    best_path = None
    best_len = 0
    
    # Try multiple random orderings to find longer paths
    for trial in range(20):
        path = [start]
        used = set()
        visited = {start}
        
        while True:
            curr = path[-1]
            # Get available neighbors via remaining edges
            candidates = []
            for w in adj[curr]:
                e = tuple(sorted((curr, w)))
                if e in remaining and w not in visited:
                    candidates.append(w)
            
            if not candidates:
                break
            
            if trial == 0:
                # Deterministic: pick neighbor with highest remaining degree
                best_w = None
                best_deg = -1
                for w in candidates:
                    deg = sum(1 for x in adj[w] if tuple(sorted((w, x))) in remaining)
                    if deg > best_deg:
                        best_deg = deg
                        best_w = w
                next_v = best_w
            else:
                # Randomized with slight preference for higher degree
                weights = []
                for w in candidates:
                    deg = sum(1 for x in adj[w] if tuple(sorted((w, x))) in remaining)
                    weights.append(deg + 1)
                total = sum(weights)
                r = random.random() * total
                cum = 0
                next_v = candidates[-1]
                for w, wt in zip(candidates, weights):
                    cum += wt
                    if r <= cum:
                        next_v = w
                        break
            
            path.append(next_v)
            visited.add(next_v)
            e = tuple(sorted((curr, next_v)))
            used.add(e)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
            best_used = used
    
    if best_path is None or len(best_path) < 2:
        # Fallback: just one edge
        for w in adj[start]:
            e = tuple(sorted((start, w)))
            if e in remaining:
                return [start, w], {e}
        return [], set()
    
    return best_path, best_used


def _decompose_flow(edges, n, deadline):
    """
    Iteratively extract paths from the residual graph.
    Each iteration: pick a start vertex, find a long path, remove its edges.
    Uses a priority: start from vertices with highest remaining degree.
    """
    remaining = set(edges)
    adj = _build_adj(edges)
    paths = []
    
    while remaining and time.time() < deadline:
        # Pick start vertex: highest remaining degree
        deg_map = {}
        for v in range(n):
            deg = sum(1 for w in adj[v] if tuple(sorted((v, w))) in remaining)
            if deg > 0:
                deg_map[v] = deg
        
        if not deg_map:
            break
        
        # Pick vertex with max degree (ties broken randomly for diversity)
        max_deg = max(deg_map.values())
        candidates = [v for v, d in deg_map.items() if d == max_deg]
        start = random.choice(candidates)
        
        path, used = _greedy_longest_path(start, adj, remaining, n)
        
        if not path or len(path) < 2:
            # Try other vertices
            found = False
            for v in sorted(deg_map, key=lambda x: -deg_map[x]):
                path, used = _greedy_longest_path(v, adj, remaining, n)
                if path and len(path) >= 2:
                    found = True
                    break
            if not found:
                break
        
        paths.append(path)
        remaining -= used
    
    return paths


def _merge_paths(paths, edges_set, n):
    """
    Try to merge pairs of paths by finding an unused edge connecting endpoints.
    Returns merged list of paths.
    """
    if len(paths) < 2:
        return paths
    
    # Compute used edges
    used = set()
    for p in paths:
        for i in range(len(p) - 1):
            used.add(tuple(sorted((p[i], p[i+1]))))
    
    changed = True
    while changed:
        changed = False
        new_paths = []
        merged_any = False
        n_paths = len(paths)
        
        for i in range(n_paths):
            if i in set(range(len(new_paths))):  # already merged into new_paths
                continue
            pi = paths[i]
            set_i = set(pi)
            
            for j in range(i + 1, n_paths):
                if any(k == j for k in [x for x in range(len(new_paths))]):
                    continue
                pj = paths[j]
                set_j = set(pj)
                
                # Try all 4 endpoint combinations
                merged = None
                for ei, ej in [(pi[0], pj[0]), (pi[0], pj[-1]), (pi[-1], pj[0]), (pi[-1], pj[-1])]:
                    if ei in set_j or ej in set_i:
                        continue
                    e = tuple(sorted((ei, ej)))
                    if e not in edges_set or e in used:
                        continue
                    
                    # Determine merge direction
                    if ei == pi[0] and ej == pj[0]:
                        cand = list(reversed(pj)) + pi
                    elif ei == pi[0] and ej == pj[-1]:
                        cand = pj + pi
                    elif ei == pi[-1] and ej == pj[0]:
                        cand = pi + pj
                    else:  # pi[-1], pj[-1]
                        cand = pi + list(reversed(pj))
                    
                    if len(set(cand)) == len(cand):
                        merged = cand
                        break
                
                if merged:
                    new_paths.append(merged)
                    used.add(tuple(sorted((merged[0], merged[1])))) if len(merged) >= 2 else None
                    # Recompute used for the merged path
                    for k in range(len(merged) - 1):
                        used.add(tuple(sorted((merged[k], merged[k+1]))))
                    merged_any = True
                    break
            
            if not merged_any:
                new_paths.append(pi)
        
        if merged_any:
            paths = new_paths
            changed = True
        else:
            break
    
    return paths


def decompose(edges, n, deadline):
    if not edges:
        return []
    
    edges_set = set(edges)
    
    # Multi-start: try several random decompositions and keep the best
    best_paths = None
    best_count = float('inf')
    
    num_starts = 30
    per_deadline = deadline - time.time()
    per_start = max(0.5, per_deadline / num_starts)
    
    for s in range(num_starts):
        start_time = time.time()
        random.seed(s * 1000 + 42)
        paths = _decompose_flow(edges, n, start_time + per_start)
        
        # Try merging
        if paths:
            paths = _merge_paths(paths, edges_set, n)
        
        # Validate
        valid = True
        all_used = set()
        for p in paths:
            if len(p) < 2:
                continue
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e not in edges_set:
                    valid = False
                    break
                if e in all_used:
                    valid = False
                    break
                all_used.add(e)
            if not valid:
                break
        
        if valid and all_used == edges_set:
            count = len([p for p in paths if len(p) >= 2])
            if count < best_count:
                best_count = count
                best_paths = paths
        
        if time.time() > deadline - 2:
            break
    
    if best_paths is None:
        # Fallback: simple greedy
        best_paths = _decompose_flow(edges, n, deadline)
        if best_paths:
            best_paths = _merge_paths(best_paths, edges_set, n)
    
    return best_paths if best_paths else []


def construct():
    start = time.time()
    deadline = start + 140
    results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose(edges, n, deadline - 3)
        results.append(paths)
        
        if time.time() > deadline - 5:
            while len(results) < 3:
                results.append([])
            break
    
    while len(results) < 3:
        results.append([])
    
    return results
