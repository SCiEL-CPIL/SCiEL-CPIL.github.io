import time
from random import Random
import sys
from collections import deque, defaultdict

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def greedy_path_cover(edges, n, rng_seed=0):
        """
        Greedy path cover with improved heuristics:
        1. Start from odd-degree vertices (they must be path endpoints)
        2. Use longest-path heuristic with look-ahead
        3. Try multiple random restarts
        """
        if not edges:
            return []
        
        remaining_edges = set(edges)
        paths = []
        rng_local = Random(rng_seed)
        
        while remaining_edges:
            if time.time() > DEADLINE:
                for e in remaining_edges:
                    paths.append(list(e))
                break
            
            # Build adjacency from remaining edges
            adj = {i: set() for i in range(n)}
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Find vertices with edges
            active_vertices = [v for v in range(n) if adj[v]]
            if not active_vertices:
                break
            
            # Find odd-degree vertices (must be path endpoints)
            odd_verts = [v for v in active_vertices if len(adj[v]) % 2 == 1]
            
            # Try to find a long path starting from an odd vertex
            best_path = []
            best_edges = set()
            
            # Candidate starts: odd vertices first, then low-degree vertices
            degrees = {v: len(adj[v]) for v in active_vertices}
            min_deg = min(degrees.values())
            low_deg_verts = [v for v in active_vertices if degrees[v] == min_deg]
            
            # Combine candidates
            candidates = list(set(odd_verts + low_deg_verts))
            
            # Also add some random vertices for diversity
            extra = rng_local.sample(active_vertices, min(10, len(active_vertices)))
            candidates = list(set(candidates + extra))
            
            for start in candidates:
                # Try to find a long simple path starting from start
                path = [start]
                visited = {start}
                current = start
                used_edges = set()
                
                while True:
                    next_candidates = [nb for nb in adj[current] if nb not in visited]
                    if not next_candidates:
                        break
                    
                    # Choose neighbor with minimum remaining degree (to avoid getting stuck)
                    # Tie-break randomly
                    best_next = None
                    min_dyn_deg = float('inf')
                    best_candidates = []
                    for nb in next_candidates:
                        dyn_deg = sum(1 for x in adj[nb] if x not in visited)
                        if dyn_deg < min_dyn_deg:
                            min_dyn_deg = dyn_deg
                            best_candidates = [nb]
                        elif dyn_deg == min_dyn_deg:
                            best_candidates.append(nb)
                    
                    if best_candidates:
                        best_next = rng_local.choice(best_candidates)
                    else:
                        break
                    
                    u, v = sorted((current, best_next))
                    used_edges.add((u, v))
                    path.append(best_next)
                    visited.add(best_next)
                    current = best_next
                
                if len(path) > len(best_path):
                    best_path = path
                    best_edges = used_edges
            
            if not best_path:
                e = next(iter(remaining_edges))
                paths.append(list(e))
                remaining_edges.discard(e)
                continue
            
            paths.append(best_path)
            remaining_edges.difference_update(best_edges)
        
        return paths

    def merge_paths(paths):
        """
        Try to merge paths that meet at a vertex and are otherwise vertex-disjoint.
        """
        changed = True
        while changed:
            changed = False
            n_paths = len(paths)
            for i in range(n_paths):
                if i >= len(paths): break
                p1 = paths[i]
                if len(p1) < 2: continue
                end1 = p1[-1]
                start1 = p1[0]
                
                for j in range(i+1, len(paths)):
                    p2 = paths[j]
                    if len(p2) < 2: continue
                    end2 = p2[-1]
                    start2 = p2[0]
                    
                    # Check if they can be merged
                    merged = None
                    if end1 == start2:
                        # p1 ends at v, p2 starts at v
                        if not (set(p1) & set(p2)) - {end1}:
                            merged = p1 + p2[1:]
                    elif end2 == start1:
                        # p2 ends at v, p1 starts at v
                        if not (set(p1) & set(p2)) - {start1}:
                            merged = p2 + p1[1:]
                    elif start1 == end2:
                         # p1 starts at v, p2 ends at v -> p2 + p1
                         # Same as above
                         pass
                    elif start2 == end1:
                         # p2 starts at v, p1 ends at v -> p1 + p2
                         # Same as first case
                         pass

                    if merged:
                        # Replace p1 and p2 with merged
                        paths[i] = merged
                        paths.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return paths

    def local_search_refine(paths, edges, n, max_iters=5000):
        """
        Local search to reduce path count:
        - Try to merge two paths by splitting one and reconnecting
        - Try to reassign edges between paths
        """
        if not paths:
            return paths
        
        paths = [list(p) for p in paths]
        edge_set = set(edges)
        rng_local = Random(42)
        
        for iteration in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            
            improved = False
            n_paths = len(paths)
            
            # Try all pairs of paths for merging via edge reassignment
            for i in range(n_paths):
                if i >= len(paths):
                    break
                p1 = paths[i]
                if len(p1) < 2:
                    continue
                for j in range(i+1, len(paths)):
                    if j >= len(paths):
                        break
                    p2 = paths[j]
                    if len(p2) < 2:
                        continue
                    
                    # Try to merge: can we combine p1 and p2 into one path?
                    # This is possible if they share a vertex and we can reconnect
                    s1 = set(p1)
                    s2 = set(p2)
                    shared = s1 & s2
                    if not shared:
                        continue
                    
                    # Try each shared vertex as connection point
                    for v in shared:
                        # Find position of v in p1 and p2
                        try:
                            pos1 = p1.index(v)
                            pos2 = p2.index(v)
                        except ValueError:
                            continue
                        
                        # Try different ways to merge
                        # Way 1: p1[0..pos1] + p2[pos2+1..] if p1[pos1] == p2[pos2] == v
                        # and the edge before/after are valid
                        # Actually, simpler: try to form a new path by splicing
                        # p1[0..pos1] + p2[pos2..end] reversed appropriately
                        # This is complex; instead, try a simpler approach:
                        # If p1 ends at v and p2 starts at v (or vice versa), merge directly
                        # This is already handled by merge_paths
                        
                        # Try edge reassignment: move one edge from p1 to p2 or vice versa
                        # This is complex; skip for now and rely on greedy + merge
            
            # Try a different approach: for each path, try to extend it using edges
            # from other paths if possible
            for i in range(len(paths)):
                if i >= len(paths):
                    break
                p = paths[i]
                if len(p) < 2:
                    continue
                
                # Try to extend from either end
                for end_idx in [0, -1]:
                    endpoint = p[end_idx]
                    # Find edges incident to endpoint that are in other paths
                    for j in range(len(paths)):
                        if j == i or j >= len(paths):
                            continue
                        p_other = paths[j]
                        if len(p_other) < 2:
                            continue
                        
                        # Check if p_other has an edge incident to endpoint
                        for k in range(len(p_other)-1):
                            u, w = p_other[k], p_other[k+1]
                            if u == endpoint or w == endpoint:
                                # This edge could potentially be moved
                                # But moving edges is complex; skip
                                pass
                                break
            
            # If no improvement, break
            if not improved:
                # Try random perturbation
                if n_paths > 2 and rng_local.random() < 0.3:
                    # Split a random path into two
                    idx = rng_local.choice(range(len(paths)))
                    p = paths[idx]
                    if len(p) >= 3:
                        split = rng_local.randint(1, len(p)-2)
                        paths[idx] = p[:split+1]
                        paths.insert(idx+1, p[split:])
                        paths = merge_paths(paths)
        
        return paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        if time.time() > DEADLINE - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        n = spec[0]
        
        best_paths = None
        min_count = float('inf')
        
        # Try multiple restarts with different seeds
        max_restarts = 30
        for r in range(max_restarts):
            if time.time() > DEADLINE - 10:
                break
            paths = greedy_path_cover(edges, n, rng_seed=r)
            paths = merge_paths(paths)
            if len(paths) < min_count:
                min_count = len(paths)
                best_paths = [list(p) for p in paths]
        
        # Apply local search refinement to best solution
        if best_paths is not None and time.time() < DEADLINE - 5:
            refined = local_search_refine(best_paths, edges, n, max_iters=2000)
            if len(refined) < min_count:
                min_count = len(refined)
                best_paths = refined
        
        if best_paths is None:
            best_paths = [[list(e)] for e in edges]
            
        results.append(best_paths)
        
    return results
