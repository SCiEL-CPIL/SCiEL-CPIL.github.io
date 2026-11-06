"""Strong edge colourings: branch-and-bound MIS extraction + simulated annealing."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod
from collections import defaultdict
from functools import lru_cache
import math

_DEADLINE_FRAC = 0.85  # use 85% of the 150s budget for safety

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

def _build_conflict_bitmask(edges, adj, n):
    """Build conflict as bitmasks for fast operations. Returns list of ints."""
    m = len(edges)
    conflict = [0] * m
    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            if a == c or a == d or b == c or b == d:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                continue
            if c in adj[a] or c in adj[b] or d in adj[a] or d in adj[b]:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    return conflict

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_coloring_bitmask(conflict, order=None, m=None):
    """Greedy coloring using bitmask conflict. Returns list of colours."""
    if m is None:
        m = len(conflict)
    if order is None:
        order = list(range(m))
    colour = [-1] * m
    colour_used_mask = [0] * m  # for each colour, bitmask of edges using it
    
    max_possible = m
    for idx in order:
        # Find smallest colour not used by any conflicting edge
        # Build set of colours used by neighbours
        neighbour_colors = set()
        cm = conflict[idx]
        for j in range(m):
            if cm & (1 << j):
                if colour[j] >= 0:
                    neighbour_colors.add(colour[j])
        c = 0
        while c in neighbour_colors:
            c += 1
        colour[idx] = c
    
    return colour

def _greedy_coloring_fast(conflict, m, order=None):
    """Optimized greedy: precompute neighbour colour sets incrementally."""
    if order is None:
        order = list(range(m))
    colour = [-1] * m
    # colour_count[c] = number of edges with colour c
    # For each edge, track which colours its neighbours use
    neighbour_colours = [set() for _ in range(m)]
    
    for idx in order:
        # Find smallest available colour
        c = 0
        while c in neighbour_colours[idx]:
            c += 1
        colour[idx] = c
        # Update neighbours
        cm = conflict[idx]
        for j in range(m):
            if cm & (1 << j):
                neighbour_colours[j].add(c)
    
    return colour

def _count_colors(colour):
    """Return the number of distinct colours used."""
    if not colour:
        return 0
    return max(colour) + 1

def _try_recolor_bitmask(colour, idx, new_color, conflict, m):
    """Check if we can assign new_color to edge idx using bitmask."""
    cm = conflict[idx]
    for j in range(m):
        if cm & (1 << j):
            if colour[j] == new_color:
                return False
    return True

def _local_improve_bitmask(colour, conflict, m, max_iters=200):
    """Try to reduce the number of colours by recoloring edges (bitmask version)."""
    if m == 0:
        return colour
    
    current_max = max(colour) + 1
    
    for _ in range(max_iters):
        improved = False
        # Try to recolor each edge to a smaller colour
        for idx in range(m):
            current_color = colour[idx]
            for new_c in range(current_color):
                if _try_recolor_bitmask(colour, idx, new_c, conflict, m):
                    colour[idx] = new_c
                    improved = True
                    break
        
        # Try to reduce the overall max colour
        current_max = max(colour) + 1
        if current_max > 0:
            max_c = current_max - 1
            max_edges = [i for i in range(m) if colour[i] == max_c]
            for idx in max_edges:
                for new_c in range(max_c):
                    if _try_recolor_bitmask(colour, idx, new_c, conflict, m):
                        colour[idx] = new_c
                        current_max -= 1
                        improved = True
                        break
        
        if not improved:
            break
    
    return colour

def _color_merge_improve_bitmask(colour, conflict, m, max_iters=50):
    """Try to merge color classes by recoloring all edges of one color to another."""
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
                    cm = conflict[idx]
                    for j in range(m):
                        if cm & (1 << j):
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
    """Order edges by sum of endpoint degrees, largest first."""
    m = len(edges)
    edge_deg = [len(adj[u]) + len(adj[v]) for u, v in edges]
    return sorted(range(m), key=lambda i: -edge_deg[i])

def _dsatur_ordering_bitmask(conflict, m, adj, edges):
    """
    DSATUR heuristic using bitmask conflict.
    """
    if m == 0:
        return []
    
    # Precompute conflict degrees (popcount)
    conf_deg = [bin(conflict[i]).count('1') for i in range(m)]
    
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
            cm = conflict[i]
            for j in range(m):
                if cm & (1 << j):
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
        cm = conflict[best_edge]
        for j in range(m):
            if cm & (1 << j):
                if colors[j] != -1:
                    used.add(colors[j])
        c = 0
        while c in used:
            c += 1
        colors[best_edge] = c
            
    return order

def _branch_and_bound_mis(remaining_mask, conflict, m, best_size, best_set, deadline):
    """
    Branch and bound to find a large induced matching (maximal independent set in conflict graph).
    remaining_mask: bitmask of available edges
    Returns (size, set_of_edges)
    """
    if time.time() > deadline:
        return (best_size, best_set)
    
    if remaining_mask == 0:
        if len(best_set) > best_size:
            best_size = len(best_set)
            best_set = best_set[:]
        return (best_size, best_set)
    
    # Upper bound: popcount of remaining
    ub = bin(remaining_mask).count('1')
    if len(best_set) + ub <= best_size:
        return (best_size, best_set)
    
    # Pick a vertex to branch on
    # Choose vertex with highest degree in remaining for better pruning
    best_v = -1
    best_deg = -1
    tmp = remaining_mask
    while tmp:
        v = (tmp & (-tmp)).bit_length() - 1
        tmp &= tmp - 1
        # Count neighbours in remaining
        deg = bin(conflict[v] & remaining_mask).count('1')
        if deg > best_deg:
            best_deg = deg
            best_v = v
    
    if best_v < 0:
        return (best_size, best_set)
    
    # Branch 1: include best_v
    new_remaining = remaining_mask & ~(conflict[best_v] | (1 << best_v))
    best_set.append(best_v)
    best_size, best_set = _branch_and_bound_mis(new_remaining, conflict, m, best_size, best_set, deadline)
    best_set.pop()
    
    # Branch 2: exclude best_v
    new_remaining = remaining_mask & ~(1 << best_v)
    best_size, best_set = _branch_and_bound_mis(new_remaining, conflict, m, best_size, best_set, deadline)
    
    return (best_size, best_set)

def _extract_mis_bitmask(conflict, m, remaining_mask, deadline):
    """Extract a large induced matching from remaining edges using B&B MIS."""
    best_size = 0
    best_set = []
    _branch_and_bound_mis(remaining_mask, conflict, m, best_size, best_set, deadline)
    return best_set

def _mis_greedy(conflict, m, remaining_mask, seed=0):
    """Greedy MIS extraction with randomization for diversity."""
    rng = _random_mod.Random(seed)
    remaining = remaining_mask
    matching = []
    
    while remaining:
        # Pick a random edge from remaining, biased toward low-degree
        candidates = []
        tmp = remaining
        while tmp:
            v = (tmp & (-tmp)).bit_length() - 1
            tmp &= tmp - 1
            deg = bin(conflict[v] & remaining).count('1')
            candidates.append((v, deg))
        
        if not candidates:
            break
        
        # Weight inversely by degree
        weights = [1.0 / (d + 1) for _, d in candidates]
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        chosen = candidates[0][0]
        for (v, d), w in zip(candidates, weights):
            cum += w
            if r <= cum:
                chosen = v
                break
        
        matching.append(chosen)
        remaining &= ~(conflict[chosen] | (1 << chosen))
    
    return matching

def _solve_with_mis(conflict, m, deadline, seed=0):
    """
    Solve by repeatedly extracting large induced matchings via B&B MIS.
    Each MIS becomes one color class.
    """
    colour = [-1] * m
    remaining_mask = (1 << m) - 1
    color_idx = 0
    
    while remaining_mask:
        if time.time() > deadline:
            # Fallback: greedy assign remaining
            tmp = remaining_mask
            while tmp:
                e = (tmp & (-tmp)).bit_length() - 1
                tmp &= tmp - 1
                # Find smallest available colour
                used = set()
                cm = conflict[e]
                for j in range(m):
                    if cm & (1 << j):
                        if colour[j] != -1:
                            used.add(colour[j])
                c = 0
                while c in used:
                    c += 1
                colour[e] = c
            break
        
        # Extract MIS from remaining
        matching = _extract_mis_bitmask(conflict, m, remaining_mask, deadline)
        
        if not matching:
            # Fallback to single edge
            tmp = remaining_mask
            e = (tmp & (-tmp)).bit_length() - 1
            matching = [e]
        
        for e in matching:
            colour[e] = color_idx
            remaining_mask &= ~(1 << e)
        
        color_idx += 1
    
    return colour

def _simulated_annealing(colour, conflict, m, deadline, rng=None):
    """
    Simulated annealing to reduce number of colours.
    Moves: recolor an edge to a random valid colour (possibly higher).
    Objective: minimize max colour.
    """
    if rng is None:
        rng = _random_mod.Random()
    
    if m == 0:
        return colour
    
    colour = colour[:]
    current_max = max(colour) + 1
    best_colour = colour[:]
    best_max = current_max
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    while T > T_min and time.time() < deadline:
        improved = False
        # Try several moves per temperature
        for _ in range(m * 3):
            if time.time() > deadline:
                break
            
            idx = rng.randint(0, m - 1)
            current_c = colour[idx]
            
            # Choose a random target colour
            max_c_to_try = min(current_max + 2, m)
            new_c = rng.randint(0, max_c_to_try - 1)
            if new_c == current_c:
                continue
            
            # Check if valid
            if not _try_recolor_bitmask(colour, idx, new_c, conflict, m):
                continue
            
            # Apply move
            colour[idx] = new_c
            new_max = max(colour) + 1
            
            # Accept if better or with probability exp(-delta/T)
            if new_max <= current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_colour = colour[:]
                improved = True
            else:
                delta = new_max - current_max
                if rng.random() < math.exp(-delta / T):
                    current_max = new_max
                    improved = True
                else:
                    # Revert
                    colour[idx] = current_c
        
        T *= cooling
        if not improved and T < 0.5:
            break
    
    return best_colour

def _solve_instance(edges, adj, n, time_budget):
    """Solve a single instance, returning a list of colours for edges in sorted order."""
    m = len(edges)
    if m == 0:
        return []
    
    t0 = time.time()
    deadline = t0 + time_budget
    
    # Precompute conflict as bitmask
    conflict = _build_conflict_bitmask(edges, adj, n)
    
    best_colour = None
    best_count = float('inf')
    
    def _update_best(colour):
        nonlocal best_colour, best_count
        if colour is not None and all(c >= 0 for c in colour):
            count = _count_colors(colour)
            if count < best_count:
                best_count = count
                best_colour = colour[:]
    
    # Strategy 1: B&B MIS extraction (primary)
    mis_deadline = t0 + time_budget * 0.35
    for seed in range(30):
        if time.time() > mis_deadline:
            break
        colour = _solve_with_mis(conflict, m, mis_deadline, seed=seed)
        colour = _local_improve_bitmask(colour, conflict, m, max_iters=150)
        colour = _color_merge_improve_bitmask(colour, conflict, m, max_iters=40)
        _update_best(colour)
    
    # Strategy 2: DSATUR ordering
    if time.time() < t0 + time_budget * 0.5:
        order = _dsatur_ordering_bitmask(conflict, m, adj, edges)
        colour = _greedy_coloring_fast(conflict, m, order)
        colour = _local_improve_bitmask(colour, conflict, m, max_iters=200)
        colour = _color_merge_improve_bitmask(colour, conflict, m, max_iters=50)
        _update_best(colour)
    
    # Strategy 3: Largest First
    if time.time() < t0 + time_budget * 0.6:
        order = _edge_ordering_largest_first(edges, adj, n)
        colour = _greedy_coloring_fast(conflict, m, order)
        colour = _local_improve_bitmask(colour, conflict, m, max_iters=200)
        colour = _color_merge_improve_bitmask(colour, conflict, m, max_iters=50)
        _update_best(colour)
    
    # Strategy 4: Simulated annealing on best solution
    if best_colour is not None and time.time() < t0 + time_budget * 0.85:
        sa_deadline = t0 + time_budget * 0.85
        rng = _random_mod.Random(123)
        for attempt in range(5):
            if time.time() > sa_deadline:
                break
            sa_colour = _simulated_annealing(best_colour, conflict, m, sa_deadline, rng)
            _update_best(sa_colour)
    
    # Strategy 5: Randomized restarts with greedy
    if time.time() < t0 + time_budget * 0.95:
        rng = _random_mod.Random(42)
        remaining_time = time_budget - (time.time() - t0)
        max_random = max(5, int(remaining_time / 0.15))
        for attempt in range(max_random):
            if time.time() > t0 + time_budget:
                break
            order = list(range(m))
            rng.shuffle(order)
            colour = _greedy_coloring_fast(conflict, m, order)
            colour = _local_improve_bitmask(colour, conflict, m, max_iters=100)
            colour = _color_merge_improve_bitmask(colour, conflict, m, max_iters=30)
            _update_best(colour)
    
    if best_colour is None:
        best_colour = _greedy_coloring_fast(conflict, m)
    
    return best_colour

def construct():
    t_start = time.time()
    total_budget = 150.0 * _DEADLINE_FRAC
    
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    results = []
    
    # Allocate budget: first instance gets 25%, second 30%, third 45%
    # (later instances are larger and benefit more from B&B MIS)
    budget_fracs = [0.25, 0.30, 0.45]
    
    for i, spec in enumerate(specs):
        n, d, seed = spec
        edges = _bounded_degree_graph(n, d, seed)
        adj = _build_adj(edges, n)
        
        elapsed = time.time() - t_start
        remaining_total = total_budget - elapsed
        if remaining_total < 2.0:
            # Not enough time; use a very fast fallback
            conflict = _build_conflict_bitmask(edges, adj, n)
            m = len(edges)
            colour = _greedy_coloring_fast(conflict, m)
            results.append(colour)
            continue
        
        # Allocate budget based on fraction of remaining
        instance_budget = max(8.0, remaining_total * budget_fracs[i])
        colour_list = _solve_instance(edges, adj, n, instance_budget)
        results.append(colour_list)
    
    return results
# EVOLVE-BLOCK-END
