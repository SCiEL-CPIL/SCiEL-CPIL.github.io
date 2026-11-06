"""Strong edge colourings of bounded-degree graphs: DSATUR + aggressive local search with color merging."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod
from collections import defaultdict

_DEADLINE_FRAC = 0.88  # use 88% of the 150s budget

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
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_conflict(edges, adj):
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
            elif (c in adj[a]) or (d in adj[a]) or (c in adj[b]) or (d in adj[b]):
                conflict[i].add(j)
                conflict[j].add(i)
    return conflict

def _dsatur_order(conflict):
    """DSATUR ordering: pick the uncolored vertex with the most saturated (different-colored) neighbors."""
    m = len(conflict)
    color = [-1] * m
    saturated = [set() for _ in range(m)]
    order = []
    uncolored = set(range(m))
    
    for _ in range(m):
        best = None
        best_sat = -1
        best_deg = -1
        for v in uncolored:
            sat = len(saturated[v])
            deg = len(conflict[v])
            if sat > best_sat or (sat == best_sat and deg > best_deg):
                best_sat = sat
                best_deg = deg
                best = v
        order.append(best)
        uncolored.remove(best)
        # Update saturated sets for neighbors
        for u in conflict[best]:
            if u in uncolored:
                saturated[u].add(color[best])
    return order

def _greedy_coloring(conflict, order):
    """Greedy coloring in the given order."""
    m = len(conflict)
    colour = [-1] * m
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
    """Try to reduce the number of colours by recoloring edges to smaller colors."""
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
        max_c = current_max - 1
        if max_c >= 0:
            max_edges = [i for i in range(m) if colour[i] == max_c]
            for idx in max_edges:
                for new_c in range(max_c):
                    if _try_recolor(colour, idx, new_c, conflict):
                        colour[idx] = new_c
                        current_max -= 1
                        improved = True
                        break
                if current_max - 1 != max_c:
                    break
        
        if not improved:
            break
    
    return colour

def _color_merge_improve(colour, conflict, rng, max_merges=50):
    """Try to merge colour classes by recoloring a subset of edges."""
    m = len(colour)
    if m == 0:
        return colour
    
    current_max = max(colour) + 1
    if current_max <= 1:
        return colour
    
    for _ in range(max_merges):
        # Pick a colour to try to eliminate (highest first)
        target = current_max - 1
        # Find all edges with this colour
        target_edges = [i for i in range(m) if colour[i] == target]
        if not target_edges:
            current_max -= 1
            continue
        
        # Try to recolor each target edge to a lower colour
        success = True
        for idx in target_edges:
            recolored = False
            for new_c in range(target):
                if _try_recolor(colour, idx, new_c, conflict):
                    colour[idx] = new_c
                    recolored = True
                    break
            if not recolored:
                # Try swapping: recolor idx to some colour, and recolor one of its conflicting edges
                # to free up a colour
                swapped = False
                for new_c in range(target):
                    if _try_recolor(colour, idx, new_c, conflict):
                        # Check if this creates no new conflicts
                        colour[idx] = new_c
                        swapped = True
                        break
                if not swapped:
                    success = False
                    break
        
        if success:
            current_max -= 1
        else:
            # Try a random subset swap
            pass
    
    return colour

def _simulated_annealing(colour, conflict, rng, time_budget):
    """Simulated annealing to minimize the number of colours."""
    m = len(colour)
    if m == 0:
        return colour
    
    t0 = time.time()
    current = colour[:]
    current_max = max(current) + 1
    best = current[:]
    best_max = current_max
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() - t0 > time_budget:
            break
        
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Pick a random edge to recolor
        idx = rng.randrange(m)
        old_color = current[idx]
        
        # Pick a new colour (biased toward lower colours)
        if rng.random() < 0.7:
            new_color = rng.randrange(max(1, current_max))
        else:
            new_color = rng.randrange(max(1, current_max + 2))
        
        if new_color == old_color:
            continue
        
        # Check validity
        valid = True
        for j in conflict[idx]:
            if current[j] == new_color:
                valid = False
                break
        
        if valid:
            # Compute new max
            current[idx] = new_color
            new_max = max(current) + 1
            
            # Accept if better or with probability exp(-delta/T)
            if new_max <= best_max:
                if new_max < best_max:
                    best = current[:]
                    best_max = new_max
            else:
                delta = new_max - best_max
                if delta <= 0 or rng.random() < _exp_neg(delta / T):
                    pass
                else:
                    current[idx] = old_color
        else:
            # Invalid move: accept with small probability to escape local minima
            if rng.random() < 0.01 * _exp_neg(1.0 / T):
                current[idx] = new_color
            # else: keep old color
    
    # Final local improvement
    best = _local_improve(best, conflict, max_iters=50)
    return best

def _exp_neg(x):
    """Safe exp(-x) for small x."""
    if x > 50:
        return 0.0
    if x < -50:
        return 1.0
    return _math_exp(-x)

# Import math for exp
import math
_math_exp = math.exp

def _solve_instance(edges, adj, n, time_budget):
    """Solve a single instance, returning a list of colours for edges in sorted order."""
    m = len(edges)
    if m == 0:
        return []
    
    t0 = time.time()
    
    # Precompute conflict graph
    conflict = _build_conflict(edges, adj)
    
    best_colour = None
    best_count = float('inf')
    
    # Strategy 1: DSATUR ordering + greedy + local improvement
    if time.time() - t0 < time_budget * 0.3:
        order = _dsatur_order(conflict)
        colour = _greedy_coloring(conflict, order)
        colour = _local_improve(colour, conflict, max_iters=100)
        colour = _color_merge_improve(colour, conflict, _random_mod.Random(42), max_merges=50)
        count = _count_colors(colour)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
    
    # Strategy 2: Largest degree first ordering
    if time.time() - t0 < time_budget * 0.5:
        edge_degree = [len(adj[u]) + len(adj[v]) for u, v in edges]
        order = sorted(range(m), key=lambda i: -edge_degree[i])
        colour = _greedy_coloring(conflict, order)
        colour = _local_improve(colour, conflict, max_iters=100)
        colour = _color_merge_improve(colour, conflict, _random_mod.Random(43), max_merges=50)
        count = _count_colors(colour)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
    
    # Strategy 3: Random orderings
    if time.time() - t0 < time_budget * 0.7:
        rng = _random_mod.Random(44)
        for attempt in range(5):
            if time.time() - t0 > time_budget * 0.7:
                break
            order = list(range(m))
            rng.shuffle(order)
            colour = _greedy_coloring(conflict, order)
            colour = _local_improve(colour, conflict, max_iters=50)
            count = _count_colors(colour)
            if count < best_count:
                best_count = count
                best_colour = colour[:]
    
    # Strategy 4: Simulated annealing on the best solution
    remaining = time_budget - (time.time() - t0)
    if remaining > 5.0 and best_colour is not None:
        rng = _random_mod.Random(45)
        sa_colour = _simulated_annealing(best_colour, conflict, rng, remaining * 0.8)
        sa_colour = _local_improve(sa_colour, conflict, max_iters=100)
        count = _count_colors(sa_colour)
        if count < best_count:
            best_count = count
            best_colour = sa_colour[:]
    
    if best_colour is None:
        # Fallback
        order = list(range(m))
        best_colour = _greedy_coloring(conflict, order)
    
    return [best_colour[i] for i in range(m)]

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
            conflict = _build_conflict(edges, adj)
            order = list(range(len(edges)))
            colour = _greedy_coloring(conflict, order)
            results.append(colour)
            continue
        
        # Allocate budget proportional to size, but at least 15s
        instance_budget = max(15.0, remaining * 0.35)
        colour_list = _solve_instance(edges, adj, n, instance_budget)
        results.append(colour_list)
    
    return results
# EVOLVE-BLOCK-END
