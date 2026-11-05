import time
import random
from collections import defaultdict

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                continue
            # Check simple path
            if len(set(path)) != len(path):
                return False
            
            # Check edges exist and are not reused
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check all edges are covered
        if used_edges != edge_set:
            return False
        
        return True

    def greedy_standard(edges, seed):
        """Standard greedy, no target count. Used as fallback."""
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        paths = []
        
        while remaining:
            # Pick highest degree vertex in remaining graph
            degs = {}
            for u, v in remaining:
                degs[u] = degs.get(u, 0) + 1
                degs[v] = degs.get(v, 0) + 1
            if not degs:
                break
            max_d = max(degs.values())
            cands = [v for v, d in degs.items() if d == max_d]
            start_u = rng.choice(cands)
            
            neighbors = [nb for nb in adj[start_u] if (start_u, nb) in remaining or (nb, start_u) in remaining]
            if not neighbors:
                # Isolated? Should not happen if remaining is not empty and connected components are handled
                # Just pick any edge
                e = next(iter(remaining))
                start_u, start_v = e
            else:
                start_v = rng.choice(neighbors)
                
            path = [start_u, start_v]
            if (start_u, start_v) in remaining:
                remaining.remove((start_u, start_v))
            else:
                remaining.remove((start_v, start_u))
                
            # Extend
            changed = True
            while changed:
                changed = False
                # Back
                back = path[-1]
                cands = []
                for nb in adj[back]:
                    if nb in path:
                        continue
                    if (back, nb) in remaining or (nb, back) in remaining:
                        cands.append(nb)
                if cands:
                    nb = rng.choice(cands)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    path.append(nb)
                    changed = True
                    continue
                # Front
                front = path[0]
                cands = []
                for nb in adj[front]:
                    if nb in path:
                        continue
                    if (front, nb) in remaining or (nb, front) in remaining:
                        cands.append(nb)
                if cands:
                    nb = rng.choice(cands)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    path.insert(0, nb)
                    changed = True
                    continue
                    
            paths.append(path)
            
        return paths

    def try_greedy_target(edges, k, seed_val):
        """
        Attempts to cover all edges with exactly k simple paths.
        Returns the list of paths if successful, else None.
        """
        if k <= 0 and edges:
            return None
        if len(edges) < k:
            return None
            
        rng = random.Random(seed_val)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        paths = [[] for _ in range(k)]
        
        # Pick k distinct edges to start k paths - use degree heuristic
        # Prefer edges where both endpoints have high degree (more options to extend)
        edge_list = list(remaining)
        def edge_score(e):
            u, v = e
            return len(adj[u]) + len(adj[v])
        edge_list.sort(key=edge_score, reverse=True)
        
        # Take top candidates and shuffle among them for diversity
        top_count = min(len(edge_list), max(k * 3, 10))
        top_edges = edge_list[:top_count]
        rng.shuffle(top_edges)
        start_edges = top_edges[:k]
        
        for i in range(k):
            u, v = start_edges[i]
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))
            paths[i] = [u, v]
            
        # Extend paths with degree-based heuristic
        changed = True
        while changed:
            changed = False
            idxs = list(range(k))
            rng.shuffle(idxs)
            
            for i in idxs:
                if not paths[i]:
                    continue
                    
                # Try extending back
                back = paths[i][-1]
                candidates = []
                for nb in adj[back]:
                    if nb in paths[i]:
                        continue
                    if (back, nb) in remaining or (nb, back) in remaining:
                        candidates.append(nb)
                        
                if candidates:
                    # Heuristic: Pick neighbor with highest remaining degree
                    def nb_score(nb):
                        rem_deg = 0
                        for nb2 in adj[nb]:
                            if (nb, nb2) in remaining or (nb2, nb) in remaining:
                                rem_deg += 1
                        return rem_deg
                    candidates.sort(key=nb_score, reverse=True)
                    # Sometimes pick randomly from top candidates for diversity
                    top_n = min(len(candidates), 3)
                    if rng.random() < 0.3:
                        nb = rng.choice(candidates[:top_n])
                    else:
                        nb = candidates[0]
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    paths[i].append(nb)
                    changed = True
                    continue
                    
                # Try extending front
                front = paths[i][0]
                candidates = []
                for nb in adj[front]:
                    if nb in paths[i]:
                        continue
                    if (front, nb) in remaining or (nb, front) in remaining:
                        candidates.append(nb)
                        
                if candidates:
                    def nb_score(nb):
                        rem_deg = 0
                        for nb2 in adj[nb]:
                            if (nb, nb2) in remaining or (nb2, nb) in remaining:
                                rem_deg += 1
                        return rem_deg
                    candidates.sort(key=nb_score, reverse=True)
                    top_n = min(len(candidates), 3)
                    if rng.random() < 0.3:
                        nb = rng.choice(candidates[:top_n])
                    else:
                        nb = candidates[0]
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    paths[i].insert(0, nb)
                    changed = True
                    continue
                    
        if remaining:
            return None
        return paths

    def merge_paths(paths, edges):
        """
        Try to merge pairs of paths into single paths to reduce count.
        Two paths can be merged if they share a common endpoint and the other
        endpoints have an edge between them.
        """
        if len(paths) <= 1:
            return paths
        
        edge_set = set(edges)
        paths = [list(p) for p in paths]
        
        # Build endpoint index: vertex -> list of (path_idx, position 0 or 1)
        endpoint_index = defaultdict(list)
        for i, path in enumerate(paths):
            if len(path) >= 2:
                endpoint_index[path[0]].append((i, 0))
                endpoint_index[path[-1]].append((i, 1))
        
        merged = False
        new_paths = list(paths)
        
        # Try all pairs
        for i in range(len(new_paths)):
            for j in range(i + 1, len(new_paths)):
                if len(new_paths[i]) < 2 or len(new_paths[j]) < 2:
                    continue
                
                pi = new_paths[i]
                pj = new_paths[j]
                
                # Check if we can merge: pi end + pj start, or pi start + pj end, etc.
                # Case 1: pi[-1] == pj[0] -> merge as pi + pj[1:]
                if pi[-1] == pj[0]:
                    candidate = pi + pj[1:]
                    if len(set(candidate)) == len(candidate):
                        new_paths[i] = candidate
                        new_paths[j] = []
                        merged = True
                        break
                
                # Case 2: pi[-1] == pj[-1] -> merge as pi + reversed(pj)[:-1]
                if pi[-1] == pj[-1]:
                    candidate = pi + pj[-2::-1]
                    if len(set(candidate)) == len(candidate):
                        new_paths[i] = candidate
                        new_paths[j] = []
                        merged = True
                        break
                        
                # Case 3: pi[0] == pj[0] -> merge as reversed(pi)[1:] + pj
                if pi[0] == pj[0]:
                    candidate = pi[-2::-1] + pj
                    if len(set(candidate)) == len(candidate):
                        new_paths[i] = candidate
                        new_paths[j] = []
                        merged = True
                        break
                        
                # Case 4: pi[0] == pj[-1] -> merge as reversed(pi) + pj[1:]
                if pi[0] == pj[-1]:
                    candidate = pi[::-1] + pj[1:]
                    if len(set(candidate)) == len(candidate):
                        new_paths[i] = candidate
                        new_paths[j] = []
                        merged = True
                        break
                        
            if merged:
                break
                
        if merged:
            new_paths = [p for p in new_paths if len(p) >= 2]
            return merge_paths(new_paths, edges)
            
        return new_paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency and degrees
        adj = defaultdict(set)
        degrees = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        num_odd = len(odd_vertices)
        
        # Theoretical Lower Bound
        if num_odd == 0:
            lb = 1 if len(edges) == 1 else 2
        else:
            lb = num_odd // 2
            
        best_paths = None
        best_count = float('inf')
        
        # Try multiple target values: lb, lb+1, lb+2, etc.
        # Sometimes lb is impossible due to structural constraints
        max_attempts_per_target = 3000
        targets_to_try = []
        for t in range(lb, min(lb + 5, len(edges) + 1)):
            targets_to_try.append(t)
        
        for target in targets_to_try:
            if time.time() > deadline - 5:
                break
                
            attempts = 0
            while attempts < max_attempts_per_target:
                if time.time() > deadline - 5:
                    break
                attempts += 1
                paths = try_greedy_target(edges, target, seed * 10000 + target * 1000 + attempts)
                if paths:
                    # Validate
                    if validate_paths(paths, edges):
                        # Try to merge to reduce count further
                        merged = merge_paths(paths, edges)
                        if validate_paths(merged, edges):
                            if len(merged) < best_count:
                                best_count = len(merged)
                                best_paths = merged
                                if best_count == lb:
                                    return best_paths
                        else:
                            if len(paths) < best_count:
                                best_count = len(paths)
                                best_paths = paths
                                if best_count == lb:
                                    return best_paths
                        break
        
        # If greedy target failed, fallback to standard greedy (minimizing count, no target)
        if best_paths is None:
            # Try a few standard greedy runs to find a good one
            for i in range(200):
                if time.time() > deadline - 5:
                    break
                paths = greedy_standard(edges, seed * 10000 + 10000 + i)
                if validate_paths(paths, edges):
                    # Try to merge
                    merged = merge_paths(paths, edges)
                    if validate_paths(merged, edges):
                        if len(merged) < best_count:
                            best_count = len(merged)
                            best_paths = merged
                    else:
                        if len(paths) < best_count:
                            best_count = len(paths)
                            best_paths = paths
            
            if best_paths is None:
                # Last resort: single edges
                best_paths = [[u, v] for u, v in edges]
                best_count = len(best_paths)

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        # Check time before starting instance
        if time.time() > deadline - 10:
            # Not enough time, use trivial decomposition for remaining
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        paths = solve_instance(n, p, seed)
        results.append(paths)
        
    # Fallback
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
