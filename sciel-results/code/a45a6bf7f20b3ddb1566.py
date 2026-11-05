import random
import itertools
import time

def construct():
    """
    Construct a dense graph on vertices 0..19 with edge colours 0..19 such that every
    4-cycle (subgraph) has four distinct colours.
    
    Strategy:
    1. Start with K20 and a random edge coloring using colors 0..19.
    2. Use local search (simulated annealing-like) to minimize the number of 'bad' 4-cycles
       (cycles where not all 4 edges have distinct colors).
    3. If we can't reach 0 bad cycles within time, iteratively remove the edge that appears
       in the most bad cycles until we have 0 bad cycles.
    4. Return the resulting edge list.
    """
    n = 20
    n_colors = 20
    start_time = time.time()
    time_limit = 110  # Leave buffer for final processing
    
    # Precompute all 4-cycles
    # A 4-cycle on vertices a,b,c,d (a<b<c<d) can be:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d)
    # In terms of edges:
    # Cycle 1: (a,b), (b,c), (c,d), (d,a)
    # Cycle 2: (a,b), (b,d), (d,c), (c,a)
    # Cycle 3: (a,c), (c,b), (b,d), (d,a)
    
    all_edges = []
    edge_index = {}  # (u,v) -> index, with u < v
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
            edge_index[(i, j)] = len(all_edges) - 1
    
    num_edges = len(all_edges)  # 190
    
    # Precompute 4-cycles as lists of edge indices
    cycles = []  # Each cycle is a list of 4 edge indices
    for a, b, c, d in itertools.combinations(range(n), 4):
        # Cycle 1: a-b-c-d-a
        e1 = edge_index[(min(a,b), max(a,b))]
        e2 = edge_index[(min(b,c), max(b,c))]
        e3 = edge_index[(min(c,d), max(c,d))]
        e4 = edge_index[(min(d,a), max(d,a))]
        cycles.append([e1, e2, e3, e4])
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_index[(min(a,b), max(a,b))]
        e2 = edge_index[(min(b,d), max(b,d))]
        e3 = edge_index[(min(d,c), max(d,c))]
        e4 = edge_index[(min(c,a), max(c,a))]
        cycles.append([e1, e2, e3, e4])
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_index[(min(a,c), max(a,c))]
        e2 = edge_index[(min(c,b), max(c,b))]
        e3 = edge_index[(min(b,d), max(b,d))]
        e4 = edge_index[(min(d,a), max(d,a))]
        cycles.append([e1, e2, e3, e4])
    
    num_cycles = len(cycles)  # 14535
    
    # Precompute for each edge, which cycles it belongs to
    edge_to_cycles = [[] for _ in range(num_edges)]
    for i, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(i)
    
    # Initial coloring: random permutation of colors
    colors = list(range(n_colors))
    random.shuffle(colors)
    edge_colors = []
    color_idx = 0
    for i in range(num_edges):
        edge_colors.append(colors[color_idx % n_colors])
        color_idx += 1
    
    # Count bad cycles
    def count_bad_cycles(edge_colors):
        bad = 0
        for cycle in cycles:
            c0 = edge_colors[cycle[0]]
            c1 = edge_colors[cycle[1]]
            c2 = edge_colors[cycle[2]]
            c3 = edge_colors[cycle[3]]
            if len({c0, c1, c2, c3}) < 4:
                bad += 1
        return bad
    
    # Fast version using a set for distinct colors check
    def is_bad(c0, c1, c2, c3):
        return c0 == c1 or c0 == c2 or c0 == c3 or c1 == c2 or c1 == c3 or c2 == c3
    
    bad_count = count_bad_cycles(edge_colors)
    best_bad = bad_count
    best_colors = edge_colors[:]
    
    if bad_count == 0:
        # Already valid
        pass
    else:
        # Local search: try to improve by changing single edge colors
        # Use simulated annealing approach
        temperature = 1.0
        cooling = 0.999
        min_temperature = 0.01
        
        while time.time() - start_time < time_limit and bad_count > 0:
            if best_bad == 0:
                break
            if best_bad < bad_count:
                # Revert to best
                edge_colors = best_colors[:]
                bad_count = best_bad
            
            # Pick a random edge that's in at least one bad cycle
            # Or just pick any random edge
            while True:
                e = random.randint(0, num_edges - 1)
                if len(edge_to_cycles[e]) > 0:
                    break
            
            old_color = edge_colors[e]
            new_color = random.randint(0, n_colors - 1)
            if new_color == old_color:
                continue
            
            # Compute change in bad count
            delta_bad = 0
            for ci in edge_to_cycles[e]:
                cycle = cycles[ci]
                # Get current colors of other 3 edges
                other_colors = []
                for e2 in cycle:
                    if e2 != e:
                        other_colors.append(edge_colors[e2])
                
                # Before: is_bad(old_color, *other_colors)
                # After: is_bad(new_color, *other_colors)
                before = is_bad(old_color, other_colors[0], other_colors[1], other_colors[2])
                after = is_bad(new_color, other_colors[0], other_colors[1], other_colors[2])
                
                if before and not after:
                    delta_bad -= 1
                elif not before and after:
                    delta_bad += 1
            
            new_bad = bad_count + delta_bad
            
            # Accept if better or with probability exp(-delta/T)
            if new_bad <= bad_count:
                edge_colors[e] = new_color
                bad_count = new_bad
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = edge_colors[:]
            else:
                import math
                prob = math.exp(-delta_bad / temperature)
                if random.random() < prob:
                    edge_colors[e] = new_color
                    bad_count = new_bad
            
            temperature *= cooling
            if temperature < min_temperature:
                temperature = min_temperature
        
        # Use best solution found
        edge_colors = best_colors[:]
        bad_count = best_bad
    
    # If still have bad cycles, remove edges to eliminate them
    while bad_count > 0 and time.time() - start_time < time_limit - 5:
        # Find the edge that appears in the most bad cycles
        # First, identify which cycles are bad
        bad_cycles_set = set()
        for i, cycle in enumerate(cycles):
            c0 = edge_colors[cycle[0]]
            c1 = edge_colors[cycle[1]]
            c2 = edge_colors[cycle[2]]
            c3 = edge_colors[cycle[3]]
            if len({c0, c1, c2, c3}) < 4:
                bad_cycles_set.add(i)
        
        bad_count = len(bad_cycles_set)
        if bad_count == 0:
            break
        
        # Count how many bad cycles each edge is in
        edge_bad_count = [0] * num_edges
        for ci in bad_cycles_set:
            for e in cycles[ci]:
                edge_bad_count[e] += 1
        
        # Find edge with max bad cycle count
        max_edge = -1
        max_count = 0
        for e in range(num_edges):
            if edge_bad_count[e] > max_count:
                max_count = edge_bad_count[e]
                max_edge = e
        
        if max_edge == -1 or max_count == 0:
            break
        
        # Remove this edge by marking it as removed
        # We'll track removed edges
        if 'removed_edges' not in locals():
            removed_edges = set()
        
        removed_edges.add(max_edge)
        
        # Recalculate bad cycles
        bad_count = 0
        bad_cycles_set = set()
        for i, cycle in enumerate(cycles):
            # Check if any edge in cycle is removed
            if any(e in removed_edges for e in cycle):
                continue
            c0 = edge_colors[cycle[0]]
            c1 = edge_colors[cycle[1]]
            c2 = edge_colors[cycle[2]]
            c3 = edge_colors[cycle[3]]
            if len({c0, c1, c2, c3}) < 4:
                bad_count += 1
                bad_cycles_set.add(i)
        
        if len(bad_cycles_set) == bad_count:
            break  # No progress, break to avoid infinite loop
    
    # Build final edge list
    result = []
    for i, (u, v) in enumerate(all_edges):
        if 'removed_edges' not in locals() or i not in removed_edges:
            result.append([u, v, edge_colors[i]])
    
    return result
