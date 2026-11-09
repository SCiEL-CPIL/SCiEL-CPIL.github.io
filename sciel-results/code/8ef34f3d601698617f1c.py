"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

def construct():
    # Deadline guard: finish well before the 150s wall-clock kill.
    _t0 = time.time()
    _DEADLINE = 135.0  # leave ~15s margin

    def _instance(n, d, seed):
        from random import Random
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def _build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _conflict_mask(u, v, adj):
        """Return a set of vertices that an edge in the same color class must not use as endpoints."""
        # An edge (a,b) conflicts with (u,v) if:
        # - shares vertex: a==u or a==v or b==u or b==v
        # - joined by edge: a adj u, a adj v, b adj u, b adj v
        # So forbidden endpoints = {u, v} ∪ N(u) ∪ N(v)
        s = {u, v}
        s.update(adj[u])
        s.update(adj[v])
        return s

    def _greedy_color(edges, adj, order=None):
        """Greedy first-fit coloring with conflict check."""
        n_edges = len(edges)
        if n_edges == 0:
            return []
        
        # Precompute forbidden vertex sets for each edge
        forbidden = [None] * n_edges
        for i, (u, v) in enumerate(edges):
            forbidden[i] = _conflict_mask(u, v, adj)
        
        if order is None:
            order = list(range(n_edges))
        
        colour = [None] * n_edges
        colour_to_edges = defaultdict(set)
        
        for i in order:
            u, v = edges[i]
            placed = False
            for c in range(n_edges):
                ok = True
                for j in colour_to_edges[c]:
                    a, b = edges[j]
                    if a in forbidden[i] or b in forbidden[i]:
                        ok = False
                        break
                if ok:
                    colour[i] = c
                    colour_to_edges[c].add(i)
                    placed = True
                    break
            if not placed:
                c = len(colour_to_edges)
                colour[i] = c
                colour_to_edges[c].add(i)
        
        return colour

    def _local_search(colours, edges, adj, time_limit=15.0):
        """Local search: try to move edges to lower colors, then compact."""
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colours
        
        forbidden = [None] * n_edges
        for i, (u, v) in enumerate(edges):
            forbidden[i] = _conflict_mask(u, v, adj)
        
        colour_to_edges = defaultdict(set)
        for i, c in enumerate(colours):
            colour_to_edges[c].add(i)
        
        improved = True
        while improved:
            improved = False
            if time.time() - t_start > time_limit:
                break
            for i in range(n_edges):
                cur_c = colours[i]
                for c in range(cur_c):
                    ok = True
                    for j in colour_to_edges[c]:
                        if j == i:
                            continue
                        a, b = edges[j]
                        if a in forbidden[i] or b in forbidden[i]:
                            ok = False
                            break
                    if ok:
                        colour_to_edges[cur_c].discard(i)
                        colour_to_edges[c].add(i)
                        colours[i] = c
                        improved = True
                        break
            
            # Compact colors
            present = sorted([c for c in colour_to_edges if colour_to_edges[c]])
            if present and present != list(range(len(present))):
                mapping = {old: new for new, old in enumerate(present)}
                for i in range(n_edges):
                    colours[i] = mapping[colours[i]]
                new_cte = defaultdict(set)
                for i, c in enumerate(colours):
                    new_cte[c].add(i)
                colour_to_edges = new_cte
        
        return colours

    def _swap_search(colours, edges, adj, time_limit=10.0):
        """Try swapping colors of pairs of edges to reduce total color count."""
        t_start = time.time()
        n_edges = len(edges)
        if n_edges == 0:
            return colours
        
        forbidden = [None] * n_edges
        for i, (u, v) in enumerate(edges):
            forbidden[i] = _conflict_mask(u, v, adj)
        
        colour_to_edges = defaultdict(set)
        for i, c in enumerate(colours):
            colour_to_edges[c].add(i)
        
        max_c = max(colours) if colours else 0
        
        improved = True
        while improved:
            improved = False
            if time.time() - t_start > time_limit:
                break
            
            # Try to move an edge from a higher color to a lower color
            for i in range(n_edges):
                cur_c = colours[i]
                for c in range(cur_c):
                    ok = True
                    for j in colour_to_edges[c]:
                        if j == i:
                            continue
                        a, b = edges[j]
                        if a in forbidden[i] or b in forbidden[i]:
                            ok = False
                            break
                    if ok:
                        colour_to_edges[cur_c].discard(i)
                        colour_to_edges[c].add(i)
                        colours[i] = c
                        improved = True
                        break
            
            # Compact
            present = sorted([c for c in colour_to_edges if colour_to_edges[c]])
            if present and present != list(range(len(present))):
                mapping = {old: new for new, old in enumerate(present)}
                for i in range(n_edges):
                    colours[i] = mapping[colours[i]]
                new_cte = defaultdict(set)
                for i, c in enumerate(colours):
                    new_cte[c].add(i)
                colour_to_edges = new_cte
        
        return colours

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = _instance(n, d, seed)
        adj = _build_adj(edges)
        n_edges = len(edges)
        
        # Try multiple orderings and keep the best
        best_colours = None
        best_count = n_edges + 1
        
        orderings = [
            list(range(n_edges)),  # natural
            list(range(n_edges - 1, -1, -1)),  # reverse
        ]
        
        # Degree-based ordering: sort edges by (degree[u]+degree[v]) descending
        deg = [len(adj[v]) for v in range(n)]
        deg_order = sorted(range(n_edges), key=lambda i: deg[edges[i][0]] + deg[edges[i][1]], reverse=True)
        orderings.append(deg_order)
        
        # Random orderings
        from random import Random
        rng = Random(seed + 999)
        for _ in range(3):
            ro = list(range(n_edges))
            rng.shuffle(ro)
            orderings.append(ro)
        
        for order in orderings:
            if time.time() > _DEADLINE - 30:
                break
            colours = _greedy_color(edges, adj, order)
            colours = _local_search(colours, edges, adj, time_limit=5.0)
            num_colors = max(colours) + 1 if colours else 0
            if num_colors < best_count:
                best_count = num_colors
                best_colours = list(colours)
        
        if best_colours is None:
            best_colours = _greedy_color(edges, adj)
        
        # Final local search with more time
        if time.time() < _DEADLINE - 10:
            best_colours = _local_search(best_colours, edges, adj, time_limit=10.0)
        
        out.append(best_colours)
    
    return out
# EVOLVE-BLOCK-END
