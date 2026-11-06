import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency with edge tracking
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def greedy_decompose(remaining_edges, rng):
            """Greedy: repeatedly find longest path in remaining graph."""
            remaining = set(remaining_edges)
            paths = []
            
            while remaining:
                # Build adjacency of remaining edges
                rem_adj = defaultdict(set)
                for u, v in remaining:
                    rem_adj[u].add(v)
                    rem_adj[v].add(u)
                
                # Find all vertices with remaining edges
                active = set()
                for u, v in remaining:
                    active.add(u)
                    active.add(v)
                
                best_path = []
                
                # Try each active vertex as start, with multiple randomizations
                nodes_list = list(active)
                rng.shuffle(nodes_list)
                
                for start in nodes_list:
                    # Greedy extend from start
                    path = [start]
                    visited = {start}
                    cur = start
                    
                    while True:
                        # Find best next vertex: prefer low degree to keep options open
                        cands = []
                        for nb in rem_adj[cur]:
                            if nb not in visited:
                                # Count degree of nb in remaining graph
                                deg = len(rem_adj[nb])
                                cands.append((deg, rng.random(), nb))
                        
                        if not cands:
                            break
                        
                        # Sort by degree (low first), then random
                        cands.sort(key=lambda x: (x[0], x[1]))
                        next_v = cands[0][2]
                        
                        edge = tuple(sorted((cur, next_v)))
                        if edge not in remaining:
                            break
                        
                        path.append(next_v)
                        visited.add(next_v)
                        remaining.discard(edge)
                        cur = next_v
                    
                    if len(path) > len(best_path):
                        best_path = path
                
                if best_path:
                    paths.append(best_path)
                else:
                    # Should not happen if remaining is non-empty
                    break
            
            return paths

        def merge_paths(paths, edges_set, rng):
            """Try to merge pairs of paths to reduce count."""
            paths = [list(p) for p in paths]
            changed = True
            while changed:
                changed = False
                for i in range(len(paths)):
                    if len(paths[i]) < 2:
                        continue
                    for j in range(i+1, len(paths)):
                        if len(paths[j]) < 2:
                            continue
                        # Check if we can merge path i and path j
                        # They can merge if an endpoint of i is adjacent to an endpoint of j
                        # via an edge not used in either path
                        pi = paths[i]
                        pj = paths[j]
                        
                        # Edges used in pi
                        edges_pi = set()
                        for k in range(len(pi)-1):
                            edges_pi.add(tuple(sorted((pi[k], pi[k+1]))))
                        # Edges used in pj
                        edges_pj = set()
                        for k in range(len(pj)-1):
                            edges_pj.add(tuple(sorted((pj[k], pj[k+1]))))
                        
                        # Try all combinations of endpoints
                        for end_i in [pi[0], pi[-1]]:
                            for end_j in [pj[0], pj[-1]]:
                                if end_i == end_j:
                                    continue
                                edge = tuple(sorted((end_i, end_j)))
                                if edge in edges_set and edge not in edges_pi and edge not in edges_pj:
                                    # Check if merging creates a simple path
                                    # Merge: reverse pi if needed, then append pj
                                    # We need the shared edge to connect an endpoint of pi to an endpoint of pj
                                    # If we connect pi's end_i to pj's end_j, the merged path is:
                                    # (pi from start to end_i) + (edge) + (pj from end_j to end)
                                    # But we need to check simplicity: no repeated vertices
                                    
                                    # Determine orientation
                                    # pi goes from pi[0] to pi[-1]. If end_i == pi[-1], pi is as is.
                                    # Otherwise, reverse pi.
                                    if end_i == pi[0]:
                                        new_pi = pi[::-1]
                                    else:
                                        new_pi = pi
                                    
                                    if end_j == pj[-1]:
                                        new_pj = pj
                                    else:
                                        new_pj = pj[::-1]
                                    
                                    # Now new_pi ends at end_i, new_pj starts at end_j
                                    # Merged path: new_pi + new_pj[1:]
                                    merged = new_pi + new_pj[1:]
                                    
                                    # Check simplicity
                                    if len(set(merged)) == len(merged):
                                        # Valid merge
                                        paths[i] = merged
                                        paths[j] = []
                                        changed = True
                                        break
                            if changed:
                                break
                        if changed:
                            break
                
                # Remove empty paths
                paths = [p for p in paths if len(p) >= 2]
            
            return paths

        def local_search(paths, edges_set, rng, max_iters=50):
            """Local search: try to improve by splitting and re-merging."""
            paths = [list(p) for p in paths]
            best = len(paths)
            
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Try a random split-and-remerge
                if len(paths) < 2:
                    break
                
                i = rng.randrange(len(paths))
                j = rng.randrange(len(paths))
                if i == j:
                    continue
                
                pi = list(paths[i])
                pj = list(paths[j])
                
                # Try to split pi at a random point and re-merge with pj
                if len(pi) > 2:
                    split = rng.randrange(1, len(pi)-1)
                    part1 = pi[:split+1]
                    part2 = pi[split:]
                    
                    # Try to merge part1 with pj
                    new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                    
                    # Try merging part1 and pj
                    merged12 = None
                    for e1 in [part1[0], part1[-1]]:
                        for e2 in [pj[0], pj[-1]]:
                            if e1 == e2:
                                continue
                            edge = tuple(sorted((e1, e2)))
                            if edge in edges_set:
                                # Check edges
                                edges_p1 = set()
                                for k in range(len(part1)-1):
                                    edges_p1.add(tuple(sorted((part1[k], part1[k+1]))))
                                edges_pj2 = set()
                                for k in range(len(pj)-1):
                                    edges_pj2.add(tuple(sorted((pj[k], pj[k+1]))))
                                if edge not in edges_p1 and edge not in edges_pj2:
                                    if e1 == part1[0]:
                                        new_p1 = part1[::-1]
                                    else:
                                        new_p1 = part1
                                    if e2 == pj[-1]:
                                        new_pj2 = pj
                                    else:
                                        new_pj2 = pj[::-1]
                                    merged = new_p1 + new_pj2[1:]
                                    if len(set(merged)) == len(merged):
                                        merged12 = merged
                                        break
                        if merged12:
                            break
                    
                    if merged12:
                        new_paths.append(merged12)
                        new_paths.append(part2)
                        
                        if len(new_paths) < len(paths):
                            paths = new_paths
                            best = len(paths)
            
            return paths

        # Main decomposition
        edges_set = set(edges)
        rng = random.Random(42)
        
        best_paths = None
        best_count = float('inf')
        
        # Multiple greedy restarts with different random seeds
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            rng_local = random.Random(1000 + attempt)
            paths = greedy_decompose(edges_set, rng_local)
            
            # Merge paths
            paths = merge_paths(paths, edges_set, rng_local)
            
            # Local search
            if time.time() < deadline - 15:
                paths = local_search(paths, edges_set, rng_local, max_iters=20)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [list(p) for p in paths]
            
            # Early stop if we hit the theoretical lower bound
            # Lower bound: max(ceil(E/2), max_degree - 1) roughly, but for simplicity use E - max_matching
            if best_count <= 10:
                break
        
        if best_paths is None:
            # Fallback: each edge is its own path
            best_paths = [[u, v] for u, v in edges]
        
        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        paths = decompose(edges)
        if not paths:
            paths = [[u, v] for u, v in edges]
        results.append(paths)
    
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
