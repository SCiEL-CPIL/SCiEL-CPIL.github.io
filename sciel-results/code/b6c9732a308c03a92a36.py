"""Strong edge colouring via DSATUR + local improvement on the conflict graph."""

import time
import random

def construct():
    deadline = time.time() + 130  # safety margin under 150s kill
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict(edges, adj):
        m = len(edges)
        # Map each vertex to set of incident edge indices
        vert_edges = {}
        for i, (u, v) in enumerate(edges):
            vert_edges.setdefault(u, []).append(i)
            vert_edges.setdefault(v, []).append(i)
        
        # Conflict: two edges conflict if they share a vertex OR are joined by an edge
        # i.e., edges (a,b) and (c,d) conflict if {a,b} ∩ {c,d} ≠ ∅ OR 
        #    (a,c) in adj OR (a,d) in adj OR (b,c) in adj OR (b,d) in adj
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i+1, m):
                u2, v2 = edges[j]
                # Check if they share a vertex
                if u1 in (u2, v2) or v1 in (u2, v2):
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if joined by an edge
                if u2 in adj.get(u1, set()) or u2 in adj.get(v1, set()) or \
                   v2 in adj.get(u1, set()) or v2 in adj.get(v1, set()):
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict
    
    def dsatur_coloring(conflict, m):
        """Color using DSATUR heuristic."""
        color = [-1] * m
        # Track saturation: number of distinct colors in neighborhood
        sat = [set() for _ in range(m)]
        # Order vertices by degree descending as tiebreaker
        degrees = sorted(range(m), key=lambda i: (-len(conflict[i]), i))
        uncolored = set(range(m))
        
        for _ in range(m):
            # Pick uncolored vertex with max saturation, then max degree
            best_v = None
            best_key = None
            for v in uncolored:
                key = (len(sat[v]), -len(conflict[v]), v)
                if best_key is None or key > best_key:
                    best_key = key
                    best_v = v
            
            # Assign smallest available color
            used = sat[best_v]
            c = 0
            while c in used:
                c += 1
            color[best_v] = c
            uncolored.remove(best_v)
            
            # Update saturation of neighbors
            for u in conflict[best_v]:
                if u in uncolored:
                    sat[u].add(c)
        
        return color, max(color) + 1
    
    def polish(color, conflict, m, max_iters=50):
        """Try to reduce colors by reassigning vertices to lower colors."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for v in range(m):
                used = set()
                for u in conflict[v]:
                    if color[u] != -1:
                        used.add(color[u])
                cur = color[v]
                for c in range(cur):
                    if c not in used:
                        color[v] = c
                        improved = True
                        break
        return color
    
    def try_reduce_k(color, conflict, m, K, timeout=5.0):
        """Try to recolor with at most K colors using backtracking."""
        start = time.time()
        # Greedy recolor attempt with limited colors
        new_color = [-1] * m
        # Order by DSATUR-like priority
        sat = [set() for _ in range(m)]
        uncolored = list(range(m))
        
        # Use a simple greedy with backtracking limit
        # For small K reduction, just try greedy
        for v in sorted(uncolored, key=lambda i: (-len(conflict[i]), i)):
            used = set()
            for u in conflict[v]:
                if new_color[u] != -1:
                    used.add(new_color[u])
            c = 0
            while c < K and c in used:
                c += 1
            if c < K:
                new_color[v] = c
            else:
                return None  # Failed
        return new_color
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 5:
            # Not enough time; use greedy for remaining
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            # Simple greedy coloring
            conflict = build_conflict(edges, adj)
            m = len(edges)
            color = [-1] * m
            for i in range(m):
                used = set()
                for j in conflict[i]:
                    if color[j] != -1:
                        used.add(color[j])
                c = 0
                while c in used:
                    c += 1
                color[i] = c
            out.append(color)
            continue
        
        edges = instance(*spec)
        n, d = spec[0], spec[1]
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        conflict = build_conflict(edges, adj)
        m = len(edges)
        
        # Initial coloring with DSATUR
        color, K = dsatur_coloring(conflict, m)
        
        # Polish
        color = polish(color, conflict, m)
        K = max(color) + 1 if color else 0
        
        # Try to reduce K
        if time.time() < deadline - 10:
            K_try = K - 1
            while K_try >= 1 and time.time() < deadline - 5:
                new_color = try_reduce_k(color, conflict, m, K_try, timeout=3.0)
                if new_color is not None:
                    color = new_color
                    K = K_try
                    color = polish(color, conflict, m)
                    K = max(color) + 1 if color else 0
                    K_try = K - 1
                else:
                    break
        
        out.append(color)
    
    return out
