"""Strong edge colourings of bounded-degree graphs: Peeling-based Induced Matching + Local Search."""
# EVOLVE-BLOCK-START
import time
import random as _random_mod
from collections import defaultdict

_DEADLINE_FRAC = 0.90  # Use 90% of the 150s budget

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

def _dsatur_order(conflict):
    """DSATUR ordering."""
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
        for u in conflict[best]:
            if u in uncolored:
                saturated[u].add(color[best])
    return order

def _peeling_coloring(conflict, rng, max_iterations=100):
    """
    Structural strategy: Iteratively extract a maximal induced matching for the next color.
    1. Start with all edges uncolored.
    2. For color k, find a set of edges that form an induced matching.
    3. Assign color k to these edges.
    4. Remove these edges from the graph.
    5. Repeat.
    """
    m = len(conflict)
    if m == 0:
        return []
    
    colour = [-1] * m
    remaining = set(range(m))
    current_color = 0
    
    while remaining:
        if current_color > max_iterations:
            break
            
        # Find a maximal induced matching in the subgraph induced by 'remaining'
        # We use a greedy approach: pick an edge, add to matching, remove its neighbors from consideration
        matching = []
        available = set(remaining)
        
        # Order edges in 'available' by degree in the conflict graph (restricted to available)
        # To make it randomized, we can shuffle or use random tie-breaking
        # For simplicity and speed, we pick edges with high "local density" first
        # But to ensure diversity across restarts, we use rng to break ties or choose starting edge
        
        # Heuristic: Pick the edge with the highest number of conflicts in 'available'
        # This tends to pick "central" edges, leaving "peripheral" ones for later?
        # Actually, for induced matching, we want edges that are far apart.
        # A better heuristic for induced matching: pick an edge, then remove its "forbidden" neighbors.
        
        # Let's try a simple greedy:
        # 1. Shuffle available edges
        # 2. Iterate: if edge is still in available and compatible with current matching, add it.
        
        shuffled_available = list(available)
        rng.shuffle(shuffled_available)
        
        for e_idx in shuffled_available:
            if e_idx not in available:
                continue
            
            # Check if e_idx is compatible with all edges in 'matching'
            # Compatible means: no conflict (in conflict graph)
            # Wait, the conflict graph already encodes "share vertex or joined by edge".
            # So an induced matching in the original graph is just an independent set in the conflict graph.
            # Yes! Strong edge coloring is just vertex coloring of the conflict graph.
            # So "induced matching" = "independent set" in conflict graph.
            
            # So we just need a maximal independent set in the subgraph induced by 'remaining'.
            compatible = True
            for m_idx in matching:
                if e_idx in conflict[m_idx]:
                    compatible = False
                    break
            
            if compatible:
                matching.append(e_idx)
                # Remove e_idx and all its neighbors from 'available'
                # Because if we pick e_idx, we cannot pick any of its neighbors.
                # But wait, we are building a maximal independent set.
                # Standard greedy MIS:
                # Pick a vertex v. Add v to MIS. Remove v and all neighbors of v from the graph.
                
                # So we remove e_idx and conflict[e_idx] from 'available'
                to_remove = {e_idx} | conflict[e_idx]
                available -= to_remove
                
        # Assign current_color to all edges in 'matching'
        for e_idx in matching:
            colour[e_idx] = current_color
            
        # Remove matched edges from 'remaining'
        remaining -= set(matching)
        
        current_color += 1
        
    # If some edges are left (due to max_iterations or bug), color them greedily
    if remaining:
        # Fallback: greedy color the rest
        order = list(remaining)
        # We need to assign colors starting from current_color
        # But we can reuse existing colors if possible?
        # No, simpler: just assign new colors greedily to the remaining ones
        # But this might increase the max color significantly.
        # Better: use the standard greedy on the whole graph for the remaining part?
        # Actually, just assign them the next available colors.
        for e_idx in order:
            # Find smallest color not used by neighbors
            used = set()
            for j in conflict[e_idx]:
                if colour[j] != -1:
                    used.add(colour[j])
            c = 0
            while c in used:
                c += 1
            colour[e_idx] = c
            
    return colour

def _local_improve(colour, conflict, rng, time_limit):
    """
    Aggressive local search to reduce the number of colors.
    1. Try to recolor each edge to a smaller color.
    2. Try to eliminate the highest color class by moving its edges.
    3. Use swaps if direct moves fail.
    """
    m = len(colour)
    if m == 0:
        return colour
    
    t_start = time.time()
    current_max = max(colour) + 1
    
    # Precompute neighbors list for faster access
    neighbors = [list(conflict[i]) for i in range(m)]
    
    def can_recolor(idx, new_c):
        for j in neighbors[idx]:
            if colour[j] == new_c:
                return False
        return True
        
    improved = True
    while improved and (time.time() - t_start) < time_limit:
        improved = False
        current_max = max(colour) + 1
        
        # Pass 1: Try to recolor every edge to the smallest possible color
        for idx in range(m):
            old_c = colour[idx]
            for new_c in range(old_c):
                if can_recolor(idx, new_c):
                    colour[idx] = new_c
                    improved = True
                    break
        
        # Pass 2: Try to eliminate the highest color
        if current_max > 1:
            max_c = current_max - 1
            max_edges = [i for i in range(m) if colour[i] == max_c]
            
            # Try to move each edge in max_edges to a lower color
            for idx in max_edges:
                if colour[idx] != max_c:
                    continue
                moved = False
                for new_c in range(max_c):
                    if can_recolor(idx, new_c):
                        colour[idx] = new_c
                        moved = True
                        improved = True
                        break
                
                if not moved:
                    # Try a 2-swap:
                    # Find an edge j with color c1 < max_c such that:
                    # 1. idx can take color c1 (if we move j away)
                    # 2. j can take color max_c (or another color)
                    # Simplified: Try to swap colors of idx and some neighbor j?
                    # No, swapping colors doesn't help if they conflict.
                    
                    # Try to find a "chain" or just try random recoloring of neighbors?
                    # Let's try: pick a random neighbor j of idx.
                    # If we recolor j to some color c2, can idx take max_c? No, idx is max_c.
                    # We want idx to take c1.
                    # So we need to free up c1.
                    # Who has c1? Let S_c1 be the set of edges with color c1.
                    # If we move one edge k in S_c1 that conflicts with idx to another color, idx can take c1.
                    
                    # This is getting complex. Let's stick to simple recoloring first.
                    # If simple recoloring fails, we might need simulated annealing.
                    pass
    
    return colour

def _solve_instance(edges, adj, n, time_budget):
    """Solve a single instance."""
    m = len(edges)
    if m == 0:
        return []
    
    t0 = time.time()
    conflict = _build_conflict(edges, adj)
    
    best_colour = None
    best_count = float('inf')
    
    # Strategy 1: DSATUR
    if time.time() - t0 < time_budget * 0.2:
        order = _dsatur_order(conflict)
        colour = _greedy_coloring(conflict, order)
        colour = _local_improve(colour, conflict, _random_mod.Random(1), 0.5)
        count = _count_colors(colour)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
            
    # Strategy 2: Peeling (Multiple restarts)
    # Peeling is structural and often better for minimizing max color.
    rng = _random_mod.Random(42)
    attempts = 10
    for i in range(attempts):
        if time.time() - t0 > time_budget * 0.7:
            break
        colour = _peeling_coloring(conflict, rng)
        # Refine with local search
        local_time = min(0.5, (time_budget - (time.time() - t0)) * 0.5)
        if local_time > 0.1:
            colour = _local_improve(colour, conflict, rng, local_time)
        count = _count_colors(colour)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
            
    # Strategy 3: If we have time, run Simulated Annealing on the best solution
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0 and best_colour is not None:
        # Simple SA: try to reduce max color
        rng = _random_mod.Random(123)
        current = best_colour[:]
        current_max = max(current) + 1
        best_sa = current[:]
        best_sa_max = current_max
        
        T = 1.0
        for step in range(10000):
            if time.time() - t0 > time_budget * 0.95:
                break
            T = max(0.01, T * 0.99)
            
            # Pick a random edge
            idx = rng.randrange(m)
            old_c = current[idx]
            
            # Try to move to a random lower color
            if old_c > 0:
                new_c = rng.randrange(old_c)
                # Check validity
                valid = True
                for j in conflict[idx]:
                    if current[j] == new_c:
                        valid = False
                        break
                if valid:
                    current[idx] = new_c
                    new_max = max(current) + 1
                    if new_max < best_sa_max:
                        best_sa_max = new_max
                        best_sa = current[:]
                else:
                    # Reject or accept with prob exp(-delta/T)
                    # Delta is 0 if max doesn't change, 1 if max increases?
                    # Here max might decrease if old_c was the max and no other edge has it.
                    # For simplicity, only accept if valid and new_max <= best_sa_max
                    pass
                    
        if best_sa_max < best_count:
            best_count = best_sa_max
            best_colour = best_sa[:]
            
    if best_colour is None:
        # Fallback
        order = list(range(m))
        best_colour = _greedy_coloring(conflict, order)
        
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
            conflict = _build_conflict(edges, adj)
            order = list(range(len(edges)))
            colour = _greedy_coloring(conflict, order)
            results.append(colour)
            continue
        
        # Allocate budget: smaller graphs get less time, but enough for peeling
        # Peeling is fast (O(m^2) roughly). Local search is the bottleneck.
        instance_budget = max(5.0, remaining * 0.4)
        
        colour_list = _solve_instance(edges, adj, n, instance_budget)
        results.append(colour_list)
        
    return results
# EVOLVE-BLOCK-END
