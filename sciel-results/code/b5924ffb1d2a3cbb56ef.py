"""Path decompositions of three fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def min_paths_lower_bound(edges, n):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd = sum(1 for d in deg if d % 2 == 1)
        return max(odd, 2) if odd > 0 else 2

    def greedy_decompose(edges, n, seed=0):
        """Greedy: build paths by extending from odd-degree vertices first."""
        rng = random.Random(seed)
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Shuffle adjacency for variety
        for u in adj:
            rng.shuffle(adj[u])
        
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        
        used_edges = set()
        paths = []
        
        # Start from vertices with odd degree (they must be endpoints)
        odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
        rng.shuffle(odd_vertices)
        
        # For each odd vertex, try to start a path from it
        # But we need to be careful: each path has exactly 2 endpoints
        # Strategy: repeatedly find the longest unused path starting from an odd vertex
        
        remaining = list(edges)
        rng.shuffle(remaining)
        
        while remaining:
            if time.time() > deadline - 5:
                break
            
            # Pick a starting vertex: prefer odd-degree among remaining
            start_candidates = set()
            for u, v in remaining:
                start_candidates.add(u)
                start_candidates.add(v)
            
            # Prioritize odd-degree vertices
            odd_starts = [v for v in start_candidates if deg[v] % 2 == 1]
            even_starts = [v for v in start_candidates if deg[v] % 2 == 0]
            
            if odd_starts:
                rng.shuffle(odd_starts)
                start = odd_starts[0]
            else:
                rng.shuffle(even_starts)
                start = even_starts[0]
            
            # Try to extend from start using DFS-like greedy
            path = [start]
            visited_in_path = {start}
            current = start
            
            while True:
                # Find unused edges from current
                options = []
                for u, v in remaining:
                    if (u, v) in used_edges or (v, u) in used_edges:
                        continue
                    if u == current and v not in visited_in_path:
                        options.append((v, (u, v)))
                    elif v == current and u not in visited_in_path:
                        options.append((u, (u, v)))
                
                if not options:
                    break
                
                # Prefer vertices with higher remaining degree (more options)
                def rem_degree(v):
                    cnt = 0
                    for u2, v2 in remaining:
                        if (u2, v2) in used_edges or (v2, u2) in used_edges:
                            continue
                        if u2 == v or v2 == v:
                            cnt += 1
                    return cnt
                
                options.sort(key=lambda x: -rem_degree(x[0]))
                # Add some randomness
                if rng.random() < 0.3 and len(options) > 1:
                    idx = rng.randint(0, min(3, len(options) - 1))
                    next_v, edge = options[idx]
                else:
                    next_v, edge = options[0]
                
                used_edges.add(edge)
                remaining.remove(edge)
                path.append(next_v)
                visited_in_path.add(next_v)
                current = next_v
            
            paths.append(path)
        
        # If we didn't use all edges (due to time), add remaining as single-edge paths
        for u, v in remaining:
            if (u, v) not in used_edges and (v, u) not in used_edges:
                paths.append([u, v])
        
        return paths

    def try_decompose(edges, n, max_time=30):
        """Try multiple random seeds and keep the best (fewest paths)."""
        start_time = time.time()
        best_paths = None
        best_count = float('inf')
        
        for attempt in range(200):
            if time.time() - start_time > max_time:
                break
            if time.time() > deadline - 5:
                break
            
            paths = greedy_decompose(edges, n, seed=attempt)
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
            
            # Early stop if we hit the lower bound
            lb = min_paths_lower_bound(edges, n)
            if count <= lb:
                break
        
        if best_paths is None:
            # Fallback: each edge as its own path
            best_paths = [[u, v] for u, v in edges]
        
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        paths = try_decompose(edges, n, max_time=40)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
