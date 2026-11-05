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
    deadline = start_time + 140.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges):
        """
        Find minimum path decomposition using:
        1. Multiple greedy initialization strategies
        2. Local search: endpoint merging
        3. Randomized restarts with time budget
        """
        if not edges:
            return []
        
        edges_set = set(edges)
        n_vertices = max(max(u, v) for u, v in edges) + 1
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def edge_key(u, v):
            return (min(u, v), max(u, v))
        
        def path_edges(path):
            return {edge_key(path[i], path[i+1]) for i in range(len(path)-1)}
        
        def validate(paths):
            """Check if paths are valid: simple, edge-disjoint, cover all edges."""
            if not paths:
                return len(edges) == 0
            used = set()
            for path in paths:
                if len(set(path)) != len(path):
                    return False
                for i in range(len(path)-1):
                    e = edge_key(path[i], path[i+1])
                    if e not in edges_set:
                        return False
                    if e in used:
                        return False
                    used.add(e)
            return used == edges_set
        
        def greedy_init(strategy='random', rng=None):
            """
            Greedy path decomposition.
            strategy: 'random', 'deterministic', 'min_degree', 'max_degree'
            """
            if rng is None:
                rng = random.Random()
            
            remaining = set(edges)
            paths = []
            
            def get_neighbors(u):
                return [v for v in adj[u] if edge_key(u, v) in remaining]
            
            def remove_edge(u, v):
                e = edge_key(u, v)
                if e in remaining:
                    remaining.remove(e)
            
            def extend_path(path, prefer_min=True):
                """Extend path from both ends until stuck."""
                changed = True
                while changed:
                    changed = False
                    for end_pos in [rng.random() < 0.5]:
                        if end_pos:
                            cur = path[-1]
                        else:
                            cur = path[0]
                        nbrs = get_neighbors(cur)
                        valid = [n for n in nbrs if n not in path]
                        if valid:
                            if strategy == 'min_degree' or (strategy == 'random' and prefer_min):
                                nxt = min(valid, key=lambda x: len(get_neighbors(x)))
                            elif strategy == 'max_degree':
                                nxt = max(valid, key=lambda x: len(get_neighbors(x)))
                            else:
                                nxt = rng.choice(valid)
                            if end_pos:
                                path.append(nxt)
                            else:
                                path.insert(0, nxt)
                            remove_edge(cur, nxt)
                            changed = True
            
            while remaining:
                # Pick start vertex
                degs = defaultdict(int)
                for u, v in remaining:
                    degs[u] += 1
                    degs[v] += 1
                
                if strategy == 'deterministic':
                    candidates = [v for v, d in degs.items() if d == max(degs.values())]
                    start = min(candidates)
                else:
                    candidates = [v for v, d in degs.items() if d == max(degs.values())]
                    start = rng.choice(candidates)
                
                neighbors = get_neighbors(start)
                if not neighbors:
                    continue
                
                if strategy == 'deterministic':
                    end = min(neighbors)
                elif strategy == 'min_degree':
                    end = min(neighbors, key=lambda x: len(get_neighbors(x)))
                else:
                    end = rng.choice(neighbors)
                
                path = [start, end]
                remove_edge(start, end)
                extend_path(path)
                paths.append(path)
            
            return paths
        
        def try_merge(paths, i, j):
            """
            Try to merge paths[i] and paths[j] into a single path.
            Returns new list of paths if successful, else None.
            """
            p1 = paths[i]
            p2 = paths[j]
            
            e1_set = path_edges(p1)
            e2_set = path_edges(p2)
            used_edges = e1_set | e2_set
            
            # Try all pairs of endpoints
            endpoints1 = [(0, p1[0]), (len(p1)-1, p1[-1])]
            endpoints2 = [(0, p2[0]), (len(p2)-1, p2[-1])]
            
            for idx1, v1 in endpoints1:
                for idx2, v2 in endpoints2:
                    if v1 == v2:
                        continue
                    ek = edge_key(v1, v2)
                    if ek in used_edges:
                        continue
                    if v2 not in adj[v1]:
                        continue
                    
                    # Merge: connect v1 (end of p1) to v2 (end of p2)
                    # Need to orient paths so that v1 is at the end and v2 is at the start
                    p1_oriented = p1 if p1[-1] == v1 else p1[::-1]
                    p2_oriented = p2 if p2[0] == v2 else p2[::-1]
                    
                    new_path = p1_oriented + p2_oriented[1:]
                    
                    # Check simplicity
                    if len(set(new_path)) != len(new_path):
                        continue
                    
                    # Check edge validity
                    valid = True
                    for k in range(len(new_path)-1):
                        e = edge_key(new_path[k], new_path[k+1])
                        if e not in edges_set:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Valid merge
                    new_paths = [paths[k] for k in range(len(paths)) if k not in (i, j)]
                    new_paths.append(new_path)
                    return new_paths
            
            return None
        
        def local_search(paths, max_iters=2000):
            """
            Local search: repeatedly try to merge pairs of paths.
            """
            paths = list(paths)
            improved = True
            iters = 0
            
            while improved and iters < max_iters:
                improved = False
                iters += 1
                n = len(paths)
                for i in range(n):
                    for j in range(i+1, n):
                        new_paths = try_merge(paths, i, j)
                        if new_paths is not None:
                            paths = new_paths
                            improved = True
                            break
                    if improved:
                        break
            
            return paths
        
        # Best solution tracking
        best_paths = None
        best_count = float('inf')
        
        # Try multiple initialization strategies
        strategies = ['deterministic', 'min_degree', 'max_degree', 'random']
        
        for strat in strategies:
            if time.time() > deadline - 5:
                break
            
            if strat == 'deterministic':
                paths = greedy_init(strategy=strat)
            else:
                paths = greedy_init(strategy=strat, rng=random.Random(hash((tuple(edges), strat))))
            
            if not validate(paths):
                continue
            
            # Apply local search
            paths = local_search(paths)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = list(paths)
        
        # Randomized restarts
        for trial in range(200):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(hash((tuple(edges), 'trial', trial)))
            paths = greedy_init(strategy='random', rng=rng)
            
            if not validate(paths):
                continue
            
            paths = local_search(paths, max_iters=500)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = list(paths)
        
        if best_paths is None:
            # Fallback: each edge as a path
            return [[u, v] for u, v in edges]
        
        return best_paths
    
    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            if len(set(path)) != len(path):
                return False
            
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        if used_edges != edge_set:
            return False
        
        return True
    
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose(edges)
        
        # Validate
        if not validate_paths(paths, edges):
            # Fallback: each edge as a path
            paths = [[u, v] for u, v in edges]
        
        results.append(paths)
        
        if time.time() > deadline - 1:
            break
    
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
