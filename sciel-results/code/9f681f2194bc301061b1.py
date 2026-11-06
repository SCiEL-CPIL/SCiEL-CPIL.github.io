"""Strong edge colourings: iterative deepening greedy + aggressive local search."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod
from collections import defaultdict

_DEADLINE_FRAC = 0.88

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
    conflict = [set() for _ in range(m)]
    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            if a == c or a == d or b == c or b == d:
                conflict[i].add(j)
                conflict[j].add(i)
                continue
            if c in adj[a] or c in adj[b] or d in adj[a] or d in adj[b]:
                conflict[i].add(j)
                conflict[j].add(i)
    return conflict

def _greedy_coloring(edges, conflict, order=None, max_color=None):
    """Greedy coloring in the given order. Returns list of colours."""
    m = len(edges)
    if order is None:
        order = list(range(m))
    colour = [-1] * m
    
    for idx in order:
        used = set()
        for j in conflict[idx]:
            if colour[j] != -1:
                used.add(colour[j])
        c = 0
        while c in used:
            c += 1
        if max_color is not None and c >= max_color:
            return None
        colour[idx] = c
    
    return colour

def _count_colors(colour):
    if not colour:
        return 0
    return max(colour) + 1

def _try_recolor(colour, idx, new_color, conflict):
    for j in conflict[idx]:
        if colour[j] == new_color:
            return False
    return True

def _local_improve(colour, conflict, max_iters=300):
    """Aggressive local search to reduce colors."""
    m = len(colour)
    if m == 0:
        return colour
    
    current_max = max(colour) + 1
    
    for _ in range(max_iters):
        improved = False
        
        # Phase 1: Try to recolor each edge to a smaller color
        for idx in range(m):
            current_color = colour[idx]
            for new_c in range(current_color):
                if _try_recolor(colour, idx, new_c, conflict):
                    colour[idx] = new_c
                    improved = True
                    break
        
        # Phase 2: Try to eliminate the highest color
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
        
        # Phase 3: Try 2-swap moves (swap colors of two non-adjacent edges)
        if current_max > 2:
            for i in range(m):
                for j in range(i+1, m):
                    if colour[i] == colour[j]:
                        continue
                    if j in conflict[i]:
                        continue
                    ci, cj = colour[i], colour[j]
                    # Check if swap is valid
                    valid = True
                    for k in conflict[i]:
                        if colour[k] == cj and k != j:
                            valid = False
                            break
                    if valid:
                        for k in conflict[j]:
                            if colour[k] == ci and k != i:
                                valid = False
                                break
                    if valid:
                        colour[i], colour[j] = cj, ci
                        improved = True
        
        if not improved:
            break
    
    return colour

def _merge_colors(colour, conflict, max_iters=100):
    """Try to merge color classes by moving all edges of one color to another."""
    m = len(colour)
    if m == 0:
        return colour
    
    current_max = max(colour) + 1
    
    for _ in range(max_iters):
        improved = False
        
        for target_c in range(current_max - 1, -1, -1):
            if target_c >= current_max:
                continue
            edges_in_color = [i for i in range(m) if colour[i] == target_c]
            if not edges_in_color:
                continue
            
            for new_c in range(target_c):
                can_move = True
                for idx in edges_in_color:
                    for j in conflict[idx]:
                        if colour[j] == new_c:
                            can_move = False
                            break
                    if not can_move:
                        break
                if can_move:
                    for idx in edges_in_color:
                        colour[idx] = new_c
                    improved = True
                    break
            
            if improved:
                current_max = max(colour) + 1
                break
        
        if not improved:
            break
    
    return colour

def _edge_ordering_largest_first(edges, adj, n):
    m = len(edges)
    edge_deg = [len(adj[u]) + len(adj[v]) for u, v in edges]
    return sorted(range(m), key=lambda i: -edge_deg[i])

def _dsatur_ordering(edges, conflict, adj, n):
    m = len(edges)
    if m == 0:
        return []
    
    conf_deg = [len(conflict[i]) for i in range(m)]
    uncolored = set(range(m))
    order = []
    colors = [-1] * m
    
    if uncolored:
        first = max(uncolored, key=lambda i: (conf_deg[i], len(adj[edges[i][0]]) + len(adj[edges[i][1]])))
        order.append(first)
        uncolored.remove(first)
        colors[first] = 0
        
    while uncolored:
        best_edge = None
        best_sat = -1
        best_deg = -1
        
        for i in uncolored:
            sat_colors = set()
            for j in conflict[i]:
                if colors[j] != -1:
                    sat_colors.add(colors[j])
            sat = len(sat_colors)
            
            if sat > best_sat or (sat == best_sat and conf_deg[i] > best_deg):
                best_sat = sat
                best_deg = conf_deg[i]
                best_edge = i
        
        order.append(best_edge)
        uncolored.remove(best_edge)
        used = set()
        for j in conflict[best_edge]:
            if colors[j] != -1:
                used.add(colors[j])
        c = 0
        while c in used:
            c += 1
        colors[best_edge] = c
            
    return order

def _solve_by_induced_matching_peeling(edges, conflict, adj, n, seed=0, time_budget=5.0):
    """Extract induced matchings via peeling to form color classes."""
    m = len(edges)
    if m == 0:
        return []
    
    t0 = time.time()
    colour = [-1] * m
    remaining = set(range(m))
    color_idx = 0
    
    rng = _random_mod.Random(seed)
    
    while remaining:
        if time.time() - t0 > time_budget:
            for e in sorted(remaining):
                used = set()
                for c in conflict[e]:
                    if colour[c] != -1:
                        used.add(colour[c])
                c = 0
                while c in used:
                    c += 1
                colour[e] = c
            break
        
        # Pick an edge with minimum conflicts in remaining set
        best_edge = None
        best_conf = float('inf')
        for e in remaining:
            conf_count = sum(1 for c in conflict[e] if c in remaining)
            if conf_count < best_conf:
                best_conf = conf_count
                best_edge = e
        
        if best_edge is None:
            best_edge = min(remaining)
        
        # Collect all edges that don't conflict with best_edge and are in remaining
        matching = [best_edge]
        for e in remaining:
            if e == best_edge:
                continue
            if e not in conflict[best_edge]:
                # Check if e conflicts with any edge already in matching
                can_add = True
                for me in matching:
                    if e in conflict[me]:
                        can_add = False
                        break
                if can_add:
                    matching.append(e)
        
        for e in matching:
            colour[e] = color_idx
            remaining.discard(e)
        
        color_idx += 1
    
    return colour

def _solve_instance(edges, adj, n, time_budget):
    """Solve a single instance, returning a list of colours for edges in sorted order."""
    m = len(edges)
    if m == 0:
        return []
    
    t0 = time.time()
    conflict = _build_conflict(edges, adj, n)
    
    best_colour = None
    best_count = float('inf')
    
    def _update_best(colour):
        nonlocal best_colour, best_count
        if colour is not None and all(c >= 0 for c in colour):
            count = _count_colors(colour)
            if count < best_count:
                best_count = count
                best_colour = colour[:]
    
    # Strategy 1: DSATUR ordering + greedy + local search
    if time.time() - t0 < t0 + time_budget * 0.3:
        order = _dsatur_ordering(edges, conflict, adj, n)
        colour = _greedy_coloring(edges, conflict, order)
        colour = _local_improve(colour, conflict, max_iters=200)
        colour = _merge_colors(colour, conflict, max_iters=50)
        _update_best(colour)
    
    # Strategy 2: Largest-first ordering + greedy + local search
    if time.time() - t0 < t0 + time_budget * 0.4:
        order = _edge_ordering_largest_first(edges, adj, n)
        colour = _greedy_coloring(edges, conflict, order)
        colour = _local_improve(colour, conflict, max_iters=200)
        colour = _merge_colors(colour, conflict, max_iters=50)
        _update_best(colour)
    
    # Strategy 3: Induced matching peeling
    if time.time() - t0 < t0 + time_budget * 0.6:
        for seed in range(10):
            if time.time() - t0 > t0 + time_budget * 0.6:
                break
            colour = _solve_by_induced_matching_peeling(edges, conflict, adj, n, seed=seed, time_budget=time_budget * 0.03)
            colour = _local_improve(colour, conflict, max_iters=150)
            colour = _merge_colors(colour, conflict, max_iters=30)
            _update_best(colour)
    
    # Strategy 4: Randomized restarts with perturbation
    rng = _random_mod.Random(123)
    remaining_time = time_budget - (time.time() - t0)
    max_random = max(20, int(remaining_time / 0.05))
    
    for attempt in range(max_random):
        if time.time() - t0 > t0 + time_budget:
            break
        
        # Random ordering with some structure
        order = list(range(m))
        # Partially shuffle: keep some edges in DSATUR order, shuffle others
        dsatur_order = _dsatur_ordering(edges, conflict, adj, n)
        for i in range(m):
            if rng.random() < 0.5:
                order[i] = dsatur_order[i]
        
        colour = _greedy_coloring(edges, conflict, order)
        colour = _local_improve(colour, conflict, max_iters=100)
        colour = _merge_colors(colour, conflict, max_iters=30)
        _update_best(colour)
    
    if best_colour is None:
        best_colour = _greedy_coloring(edges, conflict)
    
    return best_colour

def construct():
    t_start = time.time()
    total_budget = 150.0 * _DEADLINE_FRAC
    
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    results = []
    
    budget_fracs = [0.30, 0.30, 0.40]
    
    for i, spec in enumerate(specs):
        n, d, seed = spec
        edges = _bounded_degree_graph(n, d, seed)
        adj = _build_adj(edges, n)
        
        elapsed = time.time() - t_start
        remaining_total = total_budget - elapsed
        if remaining_total < 2.0:
            conflict = _build_conflict(edges, adj, n)
            colour = _greedy_coloring(edges, conflict)
            results.append(colour)
            continue
        
        instance_budget = max(10.0, remaining_total * budget_fracs[i])
        colour_list = _solve_instance(edges, adj, n, instance_budget)
        results.append(colour_list)
    
    return results
# EVOLVE-BLOCK-END
