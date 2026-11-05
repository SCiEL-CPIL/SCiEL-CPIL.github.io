"""Program for the edge-coloring task: maximize distinct colors in K16 with no rainbow 4-cycle."""
import time
import random
import itertools

N = 16
NUM_COLORS = 120
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_index(i, j):
    """Get the index for edge (i,j) in lexicographic ordering."""
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + j - i - 1

def _precompute_cycles():
    """Precompute all 4-cycles as lists of 4 edge indices."""
    cycles = []
    for a, b, c, d in itertools.combinations(range(N), 4):
        # Three 4-cycles on {a,b,c,d}
        # Cycle 1: a-b, b-c, c-d, d-a
        cycles.append([_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(d,a)])
        # Cycle 2: a-b, b-d, d-c, c-a
        cycles.append([_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(c,a)])
        # Cycle 3: a-c, c-b, b-d, d-a
        cycles.append([_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(d,a)])
    return cycles

def _precompute_edge_cycles():
    """For each edge, precompute the list of cycles it participates in."""
    all_cycles = _precompute_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(all_cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return all_cycles, edge_to_cycles

def _check_validity(colors, all_cycles):
    """Check that no 4-cycle has four distinct colors."""
    for cyc in all_cycles:
        if len(set(colors[e] for e in cyc)) == 4:
            return False
    return True

def _is_valid_change(colors, edge_idx, new_color, all_cycles, edge_to_cycles):
    """Check if changing edge_idx to new_color keeps all cycles valid."""
    for ci in edge_to_cycles[edge_idx]:
        cyc = all_cycles[ci]
        # Get the other 3 edges in this cycle
        other_edges = [e for e in cyc if e != edge_idx]
        other_colors = set(colors[e] for e in other_edges)
        # The cycle would have colors: {new_color} | other_colors
        if len(other_colors) == 3 and new_color not in other_colors:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_coloring(all_cycles, edge_to_cycles, edge_order, color_order):
    """
    Greedy edge-coloring: process edges in edge_order, assign smallest color from color_order
    that doesn't create a rainbow 4-cycle.
    """
    colors = [None] * NUM_EDGES
    
    for e in edge_order:
        # Find the first valid color from color_order
        assigned = False
        for c in color_order:
            # Check if assigning color c to edge e creates a rainbow 4-cycle
            valid = True
            for ci in edge_to_cycles[e]:
                cyc = all_cycles[ci]
                other_edges = [x for x in cyc if x != e]
                # Only consider edges that have been colored so far
                other_colored = [x for x in other_edges if colors[x] is not None]
                if len(other_colored) < 3:
                    continue
                other_colors = set(colors[x] for x in other_colored)
                if len(other_colors) == 3 and c not in other_colors:
                    valid = False
                    break
            if valid:
                colors[e] = c
                assigned = True
                break
        if not assigned:
            # Should not happen with enough colors, but fallback
            colors[e] = color_order[0]
    
    return colors

def _simulated_annealing(colors, all_cycles, edge_to_cycles, time_limit):
    """
    Simulated Annealing to maximize the number of distinct colors.
    Maintains a valid state (no rainbow 4-cycles).
    Moves:
    1. Recolor a random edge to a random color.
    2. Swap colors of two random edges.
    Acceptance:
    - If new_count > old_count: Accept.
    - If new_count == old_count: Accept.
    - If new_count < old_count: Accept with probability exp(-(old_count - new_count) / T).
    """
    import math
    
    start_time = time.time()
    best_colors = colors[:]
    best_count = _count_colors(colors)
    current_colors = colors[:]
    current_count = best_count
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    used_colors = set(current_colors)
    all_color_list = list(range(NUM_COLORS))
    
    while time.time() - start_time < time_limit and T > T_min:
        T *= cooling_rate
        
        # Generate a move
        move_type = random.random()
        if move_type < 0.5:
            # Recolor a random edge
            e = random.randint(0, NUM_EDGES - 1)
            new_c = random.choice(all_color_list)
            old_c = current_colors[e]
            
            if new_c == old_c:
                continue
                
            # Check validity of change
            if _is_valid_change(current_colors, e, new_c, all_cycles, edge_to_cycles):
                current_colors[e] = new_c
                new_count = _count_colors(current_colors)
                
                # Update used_colors efficiently? 
                # _count_colors is O(N). We can track it, but for simplicity and correctness first, 
                # we use _count_colors. Optimization: if new_c not in used_colors, count increases by 1.
                # If old_c count becomes 0, count decreases by 1.
                
                # Let's optimize count update
                # We need to know if old_c is still present and if new_c was present.
                # This is tricky with just a set. 
                # Given N=120, _count_colors is very fast (120 ops).
                
                if new_count > current_count:
                    current_count = new_count
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = current_colors[:]
                elif new_count == current_count:
                    current_count = new_count
                else:
                    # Accept with probability
                    delta = current_count - new_count
                    if random.random() < math.exp(-delta / T):
                        current_count = new_count
                    else:
                        # Reject
                        current_colors[e] = old_c
            else:
                # Invalid move, reject
                pass
        else:
            # Swap colors of two random edges
            e1 = random.randint(0, NUM_EDGES - 1)
            e2 = random.randint(0, NUM_EDGES - 1)
            
            if e1 == e2:
                continue
                
            c1 = current_colors[e1]
            c2 = current_colors[e2]
            
            if c1 == c2:
                continue
                
            # Check validity of swap
            # We need to check if changing e1 to c2 and e2 to c1 is valid.
            # This is not simply _is_valid_change twice because the changes are interdependent.
            # However, if we temporarily set e1 to c2, check validity, then set e2 to c1, check validity...
            # Wait, _is_valid_change checks if the *current* state + change is valid.
            # If we do:
            # 1. Save old colors
            # 2. Set e1 = c2
            # 3. Check if valid (considering e2 still has c1). If invalid, revert e1.
            # 4. If valid, set e2 = c1.
            # 5. Check if valid (considering e1 has c2). If invalid, revert e2 and e1.
            
            # This sequential check works because the only interactions are through cycles containing e1 or e2.
            # A cycle containing both e1 and e2 will be checked in step 3 (e1 changed, e2 old) and step 4 (e1 new, e2 changed).
            # Actually, in step 3, the cycle sees e1=new, e2=old. In step 4, it sees e1=new, e2=new.
            # We need the final state to be valid.
            
            # Let's try:
            current_colors[e1] = c2
            valid1 = True
            for ci in edge_to_cycles[e1]:
                cyc = all_cycles[ci]
                other_edges = [x for x in cyc if x != e1]
                other_colors = set(current_colors[x] for x in other_edges)
                if len(other_colors) == 3 and c2 not in other_colors:
                    valid1 = False
                    break
            
            if valid1:
                current_colors[e2] = c1
                valid2 = True
                for ci in edge_to_cycles[e2]:
                    cyc = all_cycles[ci]
                    other_edges = [x for x in cyc if x != e2]
                    other_colors = set(current_colors[x] for x in other_edges)
                    if len(other_colors) == 3 and c1 not in other_colors:
                        valid2 = False
                        break
                
                if valid2:
                    # Both changes valid in sequence. 
                    # Note: This ensures the final state is valid.
                    new_count = _count_colors(current_colors)
                    # Swap doesn't change the set of colors, so new_count should equal current_count
                    # But just in case of logic errors or if my understanding of "distinct colors" vs "count" is off?
                    # Swap preserves the multiset of colors, so distinct count is invariant.
                    # So new_count == current_count.
                    # We accept neutral moves always.
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = current_colors[:]
                else:
                    # Revert
                    current_colors[e2] = c2
                    current_colors[e1] = c1
            else:
                # Revert
                current_colors[e1] = c1
    
    return best_colors, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline for safety (90s limit)
    
    all_cycles, edge_to_cycles = _precompute_edge_cycles()
    
    # Strategy 1: Greedy with lexicographic edge order
    edge_order_lex = list(range(NUM_EDGES))
    color_order_asc = list(range(NUM_COLORS))
    
    colors1 = _greedy_coloring(all_cycles, edge_to_cycles, edge_order_lex, color_order_asc)
    count1 = _count_colors(colors1)
    
    # Strategy 2: Greedy with reversed edge order
    edge_order_rev = list(range(NUM_EDGES - 1, -1, -1))
    colors2 = _greedy_coloring(all_cycles, edge_to_cycles, edge_order_rev, color_order_asc)
    count2 = _count_colors(colors2)
    
    # Strategy 3: Greedy with random edge order
    random.seed(42)
    edge_order_rand = list(range(NUM_EDGES))
    random.shuffle(edge_order_rand)
    colors3 = _greedy_coloring(all_cycles, edge_to_cycles, edge_order_rand, color_order_asc)
    count3 = _count_colors(colors3)
    
    # Pick the best initial solution
    best_count = max(count1, count2, count3)
    if best_count == count1:
        best_colors = colors1[:]
    elif best_count == count2:
        best_colors = colors2[:]
    else:
        best_colors = colors3[:]
    
    # Simulated Annealing phase
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_colors, best_count = _simulated_annealing(best_colors, all_cycles, edge_to_cycles, remaining_time - 5)
    
    # Verify validity
    if not _check_validity(best_colors, all_cycles):
        # Fallback to a simple valid coloring
        best_colors = [0] * NUM_EDGES
        best_count = 1
    
    return best_colors
