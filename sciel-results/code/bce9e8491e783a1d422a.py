"""Strong edge colourings of bounded-degree graphs: optimized greedy + local improvement."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod
from collections import defaultdict

_DEADLINE_FRAC = 0.82  # use 82% of the 150s budget for safety

def _bounded_degree_graph(n, d, seed):
    """Reproduce the verifier's graph generation exactly."""
    rng = _random_mod.Random(seed)
    degree = [0] * n
    edges = set()
    for _ in range(4 * n * d):
        u = rng.randrange(n)
        v = rng.randrange(n)
        e = (min(u, v), max(u, v))
        if u != v and degree[u] < d and degree[v] < d and e not in edges:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_conflict(edges, adj, n):
    """Build conflict graph: two edges conflict if they share a vertex or are joined by an edge."""
    m = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    conflict = [set() for _ in range(m)]
    
    # For each edge, find conflicting edges
    for i in range(m):
        a, b = edges[i]
        # Edges sharing a vertex
        for nb in adj[a]:
            for j in range(m):
                c, d = edges[j]
                if j == i:
                    continue
                if c == a or d == a or c == b or d == b:
                    if j > i:  # avoid duplicate
                        conflict[i].add(j)
                        conflict[j].add(i)
        # Edges joined by an edge (but not sharing a vertex)
        for x in (a, b):
            for nb in adj[x]:
                for y in (a, b):
                    if nb == x or nb == y:
                        continue
                    # Check if there's an edge (nb, y) or (nb, other) that conflicts
                    # Actually, we need to check if edge (nb, z) exists where z is adjacent to x or y
                    # Let's be more direct: for each neighbor nb of a or b,
                    # any edge (nb, w) where w is adjacent to a or b (and w != nb, w not in {a,b})
                    # conflicts with edge (a,b)
    
    # More efficient approach: for each edge (a,b), 
    # conflicting edges are:
    # 1. All edges incident to a or b
    # 2. All edges (c,d) where c is adjacent to a or b, and d is adjacent to a or b (but c,d not in {a,b})
    # Let's redo this more carefully
    
    conflict = [set() for _ in range(m)]
    
    # For each pair of edges, check if they conflict
    # But that's O(m^2). For m up to ~120, that's fine.
    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            # Share a vertex?
            if a == c or a == d or b == c or b == d:
                conflict[i].add(j)
                conflict[j].add(i)
                continue
            # Joined by an edge?
            # Check if any endpoint of edge j is adjacent to any endpoint of edge i
            if c in adj[a] or c in adj[b] or d in adj[a] or d in adj[b]:
                conflict[i].add(j)
                conflict[j].add(i)
    
    return conflict

def _greedy_coloring(edges, conflict, order=None):
    """Greedy coloring in the given order. Returns list of colours."""
    m = len(edges)
    if order is None:
        order = list(range(m))
    colour = [-1] * m
    
    # For each edge in order, find the smallest available colour
    for idx in order:
        used = set()
        for j in conflict[idx]:
            if colour[j] != -1:
                used.add(colour[j])
        c = 0
        while c in used:
            c += 1
        colour[idx] = c
    
    return colour

def _count_colors(colour):
    """Return the number of distinct colours used."""
    if not colour:
        return 0
    return max(colour) + 1

def _try_recolor(colour, idx, new_color, conflict):
    """Check if we can assign new_color to edge idx."""
    for j in conflict[idx]:
        if colour[j] == new_color:
            return False
    return True

def _local_improve(colour, conflict, max_iters=100):
    """Try to reduce the number of colours by recoloring edges."""
    m = len(colour)
    if m == 0:
        return colour
    
    current_max = max(colour) + 1
    
    for _ in range(max_iters):
        improved = False
        # Try to recolor each edge to a smaller colour
        for idx in range(m):
            current_color = colour[idx]
            for new_c in range(current_color):
                if _try_recolor(colour, idx, new_c, conflict):
                    colour[idx] = new_c
                    improved = True
                    break
        
        # Try to reduce the overall max colour
        if current_max > 0:
            max_c = current_max - 1
            max_edges = [i for i in range(m) if colour[i] == max_c]
            for idx in max_edges:
                for new_c in range(max_c):
                    if _try_recolor(colour, idx, new_c, conflict):
                        colour[idx] = new_c
                        current_max -= 1
                        improved = True
                        break
        
        if not improved:
            break
    
    return colour

def _edge_ordering_largest_first(edges, adj, n):
    """Order edges by sum of endpoint degrees, largest first."""
    m = len(edges)
    edge_deg = [len(adj[u]) + len(adj[v]) for u, v in edges]
    return sorted(range(m), key=lambda i: -edge_deg[i])

def _edge_ordering_smallest_first(edges, adj, n):
    """Order edges by sum of endpoint degrees, smallest first."""
    m = len(edges)
    edge_deg = [len(adj[u]) + len(adj[v]) for u, v in edges]
    return sorted(range(m), key=lambda i: edge_deg[i])

def _solve_instance(edges, adj, n, time_budget):
    """Solve a single instance, returning a list of colours for edges in sorted order."""
    m = len(edges)
    if m == 0:
        return []
    
    t0 = time.time()
    
    # Precompute conflict graph
    conflict = _build_conflict(edges, adj, n)
    
    best_colour = None
    best_count = float('inf')
    
    # Try deterministic orderings
    strategies = [
        ("largest_first", lambda: _edge_ordering_largest_first(edges, adj, n)),
        ("smallest_first", lambda: _edge_ordering_smallest_first(edges, adj, n)),
        ("natural", lambda: list(range(m))),
    ]
    
    for name, order_func in strategies:
        if time.time() - t0 > time_budget:
            break
        order = order_func()
        colour = _greedy_coloring(edges, conflict, order)
        colour = _local_improve(colour, conflict, max_iters=50)
        count = _count_colors(colour)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
    
    # Try random orderings if time permits
    if time.time() - t0 < time_budget * 0.6:
        rng = _random_mod.Random(12345)
        max_random = 20
        for attempt in range(max_random):
            if time.time() - t0 > time_budget:
                break
            order = list(range(m))
            rng.shuffle(order)
            colour = _greedy_coloring(edges, conflict, order)
            colour = _local_improve(colour, conflict, max_iters=30)
            count = _count_colors(colour)
            if count < best_count:
                best_count = count
                best_colour = colour[:]
    
    if best_colour is None:
        best_colour = _greedy_coloring(edges, conflict)
    
    return best_colour

def construct():
    t_start = time.time()
    total_budget = 150.0 * _DEADLINE_FRAC
    
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    results = []
    
    for spec in specs:
        n, d, seed = spec
        edges = _bounded_degree_graph(n, d, seed)
        adj = _build_adj(edges, n)
        
        remaining = total_budget - (time.time() - t_start)
        if remaining < 2.0:
            # Not enough time; use a very fast fallback
            conflict = _build_conflict(edges, adj, n)
            colour = _greedy_coloring(edges, conflict)
            results.append(colour)
            continue
        
        # Allocate budget proportional to remaining, with a minimum
        instance_budget = max(15.0, remaining * 0.35)
        colour_list = _solve_instance(edges, adj, n, instance_budget)
        results.append(colour_list)
    
    return results
# EVOLVE-BLOCK-END
