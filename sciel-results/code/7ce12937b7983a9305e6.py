# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_induced_matching(edge_set, adj, n):
        """Check if a set of edges forms an induced matching."""
        used_vertices = set()
        for u, v in edge_set:
            if u in used_vertices or v in used_vertices:
                return False
            used_vertices.add(u)
            used_vertices.add(v)
        # Check no edge connects two different edges in the matching
        for u, v in edge_set:
            for w in adj[u]:
                if w != v and w in used_vertices:
                    return False
            for w in adj[v]:
                if w != u and w in used_vertices:
                    return False
        return True

    def greedy_induced_matching(uncolored_edges, adj, n):
        """Find a large induced matching from the uncolored edges using greedy."""
        # Sort edges by some heuristic: prefer edges with endpoints that have
        # fewer neighbors in the uncolored set (to reduce conflicts)
        edge_list = list(uncolored_edges)
        # Score: prefer edges whose endpoints have few neighbors in uncolored_edges
        uncolored_set = set(uncolored_edges)
        
        def edge_score(e):
            u, v = e
            # Count how many other uncolored edges are "close" to this edge
            close = 0
            for w in adj[u]:
                for x in adj[v]:
                    if (min(w, x), max(w, x)) in uncolored_set and (min(w,x),max(w,x)) != e:
                        close += 1
                    if (min(w, v), max(w, v)) in uncolored_set and (min(w,v),max(w,v)) != e:
                        close += 1
                    if (min(u, x), max(u, x)) in uncolored_set and (min(u,x),max(u,x)) != e:
                        close += 1
            return close
        
        edge_list.sort(key=edge_score)
        
        matching = []
        used_vertices = set()
        for e in edge_list:
            u, v = e
            if u in used_vertices or v in used_vertices:
                continue
            # Check if adding this edge keeps it an induced matching
            ok = True
            for w in adj[u]:
                if w != v and w in used_vertices:
                    ok = False
                    break
            if ok:
                for w in adj[v]:
                    if w != u and w in used_vertices:
                        ok = False
                        break
            if ok:
                matching.append(e)
                used_vertices.add(u)
                used_vertices.add(v)
        return matching

    def color_with_greedy_matchings(edges, adj, n):
        """Color edges by repeatedly selecting induced matchings."""
        uncolored = set(edges)
        color_of = {}
        next_color = 0
        while uncolored:
            matching = greedy_induced_matching(uncolored, adj, n)
            if not matching:
                # Fallback: assign each remaining edge its own color
                for e in uncolored:
                    color_of[e] = next_color
                    next_color += 1
                break
            for e in matching:
                color_of[e] = next_color
                uncolored.discard(e)
            next_color += 1
        return color_of

    def try_reduce_colors(color_of, edges, adj, n, max_iter=200):
        """Local search: try to move edges to lower-numbered colors to reduce total."""
        # Build color classes
        colors = {}
        for e, c in color_of.items():
            colors.setdefault(c, set()).add(e)
        
        # Sort colors by number of edges (try to eliminate small color classes)
        color_list = sorted(colors.keys())
        
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            for c in color_list:
                if c not in colors or len(colors[c]) == 0:
                    continue
                for e in list(colors[c]):
                    # Try to move edge e to a lower color
                    for c2 in range(c):
                        if c2 not in colors:
                            continue
                        # Check if e can be added to color class c2
                        if can_add_edge(e, colors[c2], adj, n):
                            colors[c].discard(e)
                            colors[c2].add(e)
                            color_of[e] = c2
                            improved = True
                            break
                    if improved:
                        break
                # Remove empty color class
                if c in colors and len(colors[c]) == 0:
                    del colors[c]
        
        # Renumber colors to be contiguous
        old_to_new = {}
        for i, c in enumerate(sorted(colors.keys())):
            old_to_new[c] = i
        new_color_of = {}
        for e, c in color_of.items():
            new_color_of[e] = old_to_new[c]
        return new_color_of

    def can_add_edge(edge, color_class, adj, n):
        """Check if edge can be added to an existing color class (induced matching)."""
        u, v = edge
        used = set()
        for a, b in color_class:
            used.add(a)
            used.add(b)
        if u in used or v in used:
            return False
        for w in adj[u]:
            if w != v and w in used:
                return False
        for w in adj[v]:
            if w != u and w in used:
                return False
        return True

    def local_search_swap(color_of, edges, adj, n, max_iter=500):
        """Try swaps between two color classes to enable further reductions."""
        colors = {}
        for e, c in color_of.items():
            colors.setdefault(c, set()).add(e)
        
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            c_list = sorted(colors.keys())
            for i, c1 in enumerate(c_list):
                if c1 not in colors:
                    continue
                for c2 in c_list[i+1:]:
                    if c2 not in colors:
                        continue
                    # Try swapping one edge from c1 with one from c2
                    for e1 in list(colors[c1]):
                        for e2 in list(colors[c2]):
                            # Remove e1 from c1, e2 from c2
                            new_c1 = colors[c1] - {e1}
                            new_c2 = colors[c2] - {e2}
                            # Check if e2 fits in c1 and e1 fits in c2
                            if can_add_edge(e2, new_c1, adj, n) and can_add_edge(e1, new_c2, adj, n):
                                colors[c1] = new_c1 | {e2}
                                colors[c2] = new_c2 | {e1}
                                color_of[e1] = c2
                                color_of[e2] = c1
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            # After swaps, try to reduce again
            if improved:
                color_of = try_reduce_colors(color_of, edges, adj, n, max_iter=100)
                # Rebuild colors
                colors = {}
                for e, c in color_of.items():
                    colors.setdefault(c, set()).add(e)
        
        # Renumber
        old_to_new = {}
        for i, c in enumerate(sorted(colors.keys())):
            old_to_new[c] = i
        new_color_of = {}
        for e, c in color_of.items():
            new_color_of[e] = old_to_new[c]
        return new_color_of

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE:
            # Return best we have so far with a trivial coloring for remaining
            while len(out) < 3:
                n, d, seed = spec
                edges = instance(n, d, seed)
                adj = build_adj(edges, n)
                color_of = {}
                for i, e in enumerate(edges):
                    color_of[e] = i
                out.append([color_of[e] for e in edges])
            return out
        
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = build_adj(edges, n)
        
        # Phase 1: Greedy induced matching coloring
        color_of = color_with_greedy_matchings(edges, adj, n)
        num_colors = max(color_of.values()) + 1 if color_of else 0
        
        # Phase 2: Local search to reduce colors
        if time.time() < DEADLINE - 5:
            color_of = try_reduce_colors(color_of, edges, adj, n)
            num_colors = max(color_of.values()) + 1 if color_of else 0
        
        # Phase 3: Swap-based local search
        if time.time() < DEADLINE - 3:
            color_of = local_search_swap(color_of, edges, adj, n)
            num_colors = max(color_of.values()) + 1 if color_of else 0
        
        out.append([color_of[e] for e in edges])
    
    return out
# EVOLVE-BLOCK-END
