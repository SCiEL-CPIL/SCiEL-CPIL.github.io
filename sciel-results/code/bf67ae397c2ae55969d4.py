"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # 120s wall-clock guard with safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_greedy(edges):
        """Decompose edge set into edge-disjoint simple paths using greedy approach."""
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        remaining = set(edges)
        paths = []
        
        while remaining:
            # Pick an edge to start a path
            edge = next(iter(remaining))
            u, v = edge
            remaining.discard(edge)
            
            # Try to extend the path in both directions
            path = [u, v]
            
            # Extend forward from v
            changed = True
            while changed:
                changed = False
                for w in adj[v]:
                    if (v, w) in remaining if v < w else (w, v) in remaining:
                        key = (v, w) if v < w else (w, v)
                        if key in remaining:
                            remaining.discard(key)
                            path.append(w)
                            v = w
                            changed = True
                            break
            
            # Extend backward from u
            changed = True
            while changed:
                changed = False
                for w in adj[u]:
                    key = (u, w) if u < w else (w, u)
                    if key in remaining:
                        remaining.discard(key)
                        path.insert(0, w)
                        u = w
                        changed = True
                        break
            
            paths.append(path)
        
        return paths

    def decompose_improved(edges):
        """Improved decomposition: use odd-degree vertices as path endpoints."""
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Find vertices with odd degree
        odd_vertices = set()
        for u, v in edges:
            odd_vertices.add(u)
            odd_vertices.add(v)
        # Actually need to compute degrees
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_vertices = {v for v in deg if deg[v] % 2 == 1}
        
        remaining = set(edges)
        paths = []
        
        # Start paths from odd-degree vertices (they must be endpoints)
        used_endpoints = set()
        
        # First pass: create paths starting/ending at odd vertices
        for start in sorted(odd_vertices):
            if time.time() - start_time > DEADLINE - 5:
                break
            if start in used_endpoints:
                continue
            
            # Find an unused edge incident to start
            start_edge = None
            for w in adj[start]:
                key = (start, w) if start < w else (w, start)
                if key in remaining:
                    start_edge = (start, w)
                    break
            
            if start_edge is None:
                continue
            
            u, v = start_edge
            remaining.discard(start_edge)
            path = [u, v]
            
            # Extend from v (away from u)
            cur = v
            prev = u
            while True:
                extended = False
                for w in adj[cur]:
                    if w == prev:
                        continue
                    key = (cur, w) if cur < w else (w, cur)
                    if key in remaining:
                        remaining.discard(key)
                        path.append(w)
                        prev, cur = cur, w
                        extended = True
                        break
                if not extended:
                    break
            
            # Extend backward from u (away from v)
            cur = u
            prev = v
            while True:
                extended = False
                for w in adj[cur]:
                    if w == prev:
                        continue
                    key = (cur, w) if cur < w else (w, cur)
                    if key in remaining:
                        remaining.discard(key)
                        path.insert(0, w)
                        prev, cur = cur, w
                        extended = True
                        break
                if not extended:
                    break
            
            paths.append(path)
            used_endpoints.add(start)
            # Mark the other endpoint if it's odd
            other = path[-1]
            if other in odd_vertices:
                used_endpoints.add(other)
        
        # Second pass: handle remaining edges
        while remaining:
            if time.time() - start_time > DEADLINE - 5:
                break
            edge = next(iter(remaining))
            u, v = edge
            remaining.discard(edge)
            path = [u, v]
            
            # Extend forward
            cur = v
            prev = u
            while True:
                extended = False
                for w in adj[cur]:
                    if w == prev:
                        continue
                    key = (cur, w) if cur < w else (w, cur)
                    if key in remaining:
                        remaining.discard(key)
                        path.append(w)
                        prev, cur = cur, w
                        extended = True
                        break
                if not extended:
                    break
            
            # Extend backward
            cur = u
            prev = v
            while True:
                extended = False
                for w in adj[cur]:
                    if w == prev:
                        continue
                    key = (cur, w) if cur < w else (w, cur)
                    if key in remaining:
                        remaining.discard(key)
                        path.insert(0, w)
                        prev, cur = cur, w
                        extended = True
                        break
                if not extended:
                    break
            
            paths.append(path)
        
        return paths

    def merge_paths(paths, edges):
        """Try to merge two paths if they can be combined."""
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                # Check if paths[i] and paths[j] can be merged
                p1, p2 = paths[i], paths[j]
                # p1 ends at p1[-1], p2 starts at p2[0] or vice versa
                # Check if p1[-1] == p2[0] or p1[-1] == p2[-1] etc.
                # For simple merging, we need an edge connecting endpoints
                merged = None
                for (a, b) in [
                    (p1[-1], p2[0]), (p1[-1], p2[-1]),
                    (p1[0], p2[0]), (p1[0], p2[-1])
                ]:
                    if b in adj[a]:
                        # Can connect a and b
                        if b == p2[0]:
                            merged = p1 + p2
                        elif b == p2[-1]:
                            merged = p1 + list(reversed(p2))
                        else:
                            # Need to check more carefully
                            pass
                        break
                if merged is not None and len(set(merged)) == len(merged):
                    # Verify all edges exist
                    valid = True
                    for k in range(len(merged) - 1):
                        u, v = merged[k], merged[k + 1]
                        if v not in adj[u]:
                            valid = False
                            break
                    if valid:
                        return merged, i, j
        return None, None, None

    def local_search(paths, edges, max_iters=1000):
        """Try to reduce path count by merging or rearranging."""
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE - 5:
                break
            improved = False
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    p1, p2 = paths[i], paths[j]
                    # Try all 4 ways to connect
                    for (idx1, rev1, idx2, rev2) in [(0, False, 0, False), (0, False, 1, True), (1, True, 0, False), (1, True, 1, True)]:
                        a = p1[idx1 if not rev1 else len(p1)-1-idx1]
                        b = p2[idx2 if not rev2 else len(p2)-1-idx2]
                        if b in adj[a]:
                            # Build merged path
                            seg1 = p1 if not rev1 else list(reversed(p1))
                            seg2 = p2 if not rev2 else list(reversed(p2))
                            # Find which end of seg1 connects to which end of seg2
                            if idx1 == 0 and not rev1:
                                # seg1 starts at a
                                if idx2 == 0 and not rev2:
                                    # seg2 ends at b? No, seg2 starts at b if idx2=0, not rev2
                                    # a is seg1[0], b is seg2[0] -> can't directly join
                                    pass
                                elif idx2 == 1 and rev2:
                                    # b is seg2[-1]
                                    merged = seg1 + seg2[1:]
                                else:
                                    pass
                            elif idx1 == 1 and rev1:
                                # a is seg1[-1]
                                if idx2 == 0 and not rev2:
                                    merged = seg1 + seg2[1:]
                                elif idx2 == 1 and rev2:
                                    # b is seg2[-1] -> no, that's the other end
                                    merged = seg1[:-1] + seg2
                                else:
                                    pass
                            else:
                                pass
                            
                            if 'merged' in dir() and merged is not None:
                                # Check simplicity
                                if len(set(merged)) == len(merged):
                                    # Verify edges
                                    valid = True
                                    for k in range(len(merged) - 1):
                                        u, v = merged[k], merged[k+1]
                                        if v not in adj[u]:
                                            valid = False
                                            break
                                    if valid:
                                        new_paths = paths[:i] + paths[i+1:j] + paths[j+1:] + [merged]
                                        paths = new_paths
                                        improved = True
                                        break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() - start_time > DEADLINE - 5:
            # Fallback: return remaining as individual edges
            remaining_specs = specs[specs.index(spec):]
            for s in remaining_specs:
                e = instance(*s)
                results.append([list(edge) for edge in e])
            break
        
        edges = instance(*spec)
        paths = decompose_improved(edges)
        
        # Try local search to merge paths
        if time.time() - start_time < DEADLINE - 10:
            paths = local_search(paths, edges)
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
