"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict

def construct():
    n = 20
    num_colors = 20
    
    # Precompute all edges
    edge_list = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_list.append((u, v))
            edge_to_idx[(u, v)] = idx
            idx += 1
    m = len(edge_list) # 190 edges

    # Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices. There are 3 ways to form a cycle from 4 vertices.
    cycles = []
    cycle_edges = [] # List of list of edge indices
    edge_to_cycles = defaultdict(list) # edge_idx -> list of cycle indices
    
    vertices = range(n)
    for quad in itertools.combinations(vertices, 4):
        a, b, c, d = quad
        # The 3 distinct cycles (edge sets) on these 4 vertices
        # 1. a-b-c-d-a
        e1 = [(a,b), (b,c), (c,d), (d,a)]
        # 2. a-b-d-c-a
        e2 = [(a,b), (b,d), (d,c), (c,a)]
        # 3. a-c-b-d-a
        e3 = [(a,c), (c,b), (b,d), (d,a)]
        
        for edges in [e1, e2, e3]:
            cyc_idx_set = []
            for e in edges:
                u, v = e
                if u > v:
                    u, v = v, u
                cyc_idx_set.append(edge_to_idx[(u, v)])
            
            # Unique cycle identifier (sorted tuple of edge indices)
            # Note: The 3 cycles for a quad are distinct edge sets.
            # Different quads produce different sets of 4 edges.
            # So no duplicates in the global list.
            cyc_idx = tuple(sorted(cyc_idx_set))
            cycles.append(cyc_idx)
            cycle_edges.append(cyc_idx)
            
            for e_idx in cyc_idx:
                edge_to_cycles[e_idx].append(len(cycles) - 1)

    num_cycles = len(cycles)
    
    # State variables
    present = [0] * m
    color = [0] * m
    
    # Initialize: Dense random graph
    # Start with ~90% edges to maximize potential
    for i in range(m):
        if random.random() < 0.9:
            present[i] = 1
            color[i] = random.randint(0, num_colors - 1)
        else:
            present[i] = 0
            color[i] = 0 # irrelevant if not present

    # Data structures for fast violation tracking
    # present_count[cyc_idx] = number of present edges in the cycle (0-4)
    # color_counts[cyc_idx][c] = number of edges in the cycle with color c (only for present edges)
    present_count = [0] * num_cycles
    color_counts = [[0] * num_colors for _ in range(num_cycles)]
    
    # Initialize counts
    for i in range(m):
        if present[i]:
            c = color[i]
            for cyc_idx in edge_to_cycles[i]:
                present_count[cyc_idx] += 1
                color_counts[cyc_idx][c] += 1

    # Calculate violations
    # A cycle is violated if present_count == 4 and not all colors are distinct.
    # Sum of squares of color counts > 4 implies not distinct (since 4 items, distinct means 1,1,1,1 -> sum sq 4).
    
    def is_violated(cyc_idx):
        if present_count[cyc_idx] < 4:
            return False
        counts = color_counts[cyc_idx]
        sum_sq = 0
        for cnt in counts:
            if cnt > 0:
                sum_sq += cnt * cnt
        return sum_sq > 4

    violated_cycles = set()
    for i in range(num_cycles):
        if is_violated(i):
            violated_cycles.add(i)
            
    # Objective: Maximize edges. Penalty for violations is implicit in the repair process.
    # We want to reach 0 violations with max edges.
    
    best_edges = sum(present)
    best_present = present[:]
    best_color = color[:]
    
    start_time = time.time()
    time_limit = 105.0
    
    # WalkSAT parameters
    walk_prob = 0.1
    
    while time.time() - start_time < time_limit:
        if violated_cycles:
            # Pick a random violated cycle
            cyc_idx = random.choice(list(violated_cycles))
            edges_in_cyc = cycle_edges[cyc_idx]
            
            # Try to fix by recoloring or removing
            best_move = None
            best_delta = float('inf')
            
            # Option 1: Recolor an edge
            for e_idx in edges_in_cyc:
                # Check if other 3 edges have distinct colors
                other_colors = []
                for other_e in edges_in_cyc:
                    if other_e != e_idx:
                        other_colors.append(color[other_e])
                
                if len(set(other_colors)) < 3:
                    continue # Cannot fix with single recoloring
                
                # Find valid new colors
                # New color must not be in other_colors
                valid_colors = [c for c in range(num_colors) if c not in other_colors]
                
                for new_c in valid_colors:
                    if new_c == color[e_idx]:
                        continue # No change
                    
                    # Calculate delta violations
                    # This is expensive to do exactly for every candidate.
                    # Approximation: Just pick a random valid color and check if it helps?
                    # Or, we can just apply the move and see if it reduces the set of violated cycles.
                    # Let's try a few random valid colors.
                    pass # We will implement a simpler heuristic below
            
            # Simpler Heuristic for Recoloring:
            # Try to recolor one of the edges in the violated cycle to a color that makes it rainbow.
            # Check if this recoloring creates *new* violations in other cycles containing that edge.
            
            fixed = False
            # Shuffle edges to avoid bias
            shuffled_edges = edges_in_cyc[:]
            random.shuffle(shuffled_edges)
            
            for e_idx in shuffled_edges:
                other_colors = []
                for other_e in edges_in_cyc:
                    if other_e != e_idx:
                        other_colors.append(color[other_e])
                
                if len(set(other_colors)) < 3:
                    continue
                
                valid_colors = [c for c in range(num_colors) if c not in other_colors]
                random.shuffle(valid_colors)
                
                for new_c in valid_colors:
                    if new_c == color[e_idx]:
                        continue
                    
                    # Simulate recoloring e_idx to new_c
                    old_c = color[e_idx]
                    
                    # Check impact on all cycles containing e_idx
                    new_violations = 0
                    for c_idx in edge_to_cycles[e_idx]:
                        if c_idx == cyc_idx:
                            # This cycle will be fixed (we ensured colors are distinct)
                            # So it goes from Violated to Not Violated.
                            # Delta = -1 (improvement)
                            pass
                        else:
                            # Check if this cycle becomes violated
                            # Current state: present_count[c_idx] is 4?
                            # If present_count[c_idx] < 4, it's not violated regardless of color.
                            if present_count[c_idx] < 4:
                                continue
                            
                            # present_count[c_idx] == 4
                            # We are changing color of one edge from old_c to new_c.
                            # Check if the new color distribution is distinct.
                            # We can simulate the counts.
                            counts = color_counts[c_idx]
                            # old_c count decreases by 1, new_c count increases by 1
                            # We need to check if the resulting counts are 1,1,1,1.
                            # This is equivalent to:
                            # 1. old_c count was 1 (so it becomes 0) and new_c count was 0 (so it becomes 1)?
                            #    No, we just need the final multiset to be {1,1,1,1}.
                            #    This means no color appears more than once.
                            
                            # Let's check the condition:
                            # Final count of c is:
                            # count[c] - (1 if c==old_c else 0) + (1 if c==new_c else 0)
                            
                            # A cycle is violated if any count > 1.
                            # So we check if any color has final count > 1.
                            
                            # Optimization:
                            # If old_c == new_c, no change.
                            # If old_c != new_c:
                            #   The counts change for old_c and new_c only.
                            #   Other colors are unchanged.
                            #   If any other color had count > 1, the cycle was already violated (and remains so, or was already in violated_cycles).
                            #   Wait, if the cycle was NOT in violated_cycles, it means it was valid (all counts 1) or incomplete.
                            #   If it was valid (all counts 1), then old_c count was 1, new_c count was 0 (since new_c != old_c and all are 1, new_c wasn't present? No, if all 4 are distinct, new_c is not in the set).
                            #   So if we change old_c to new_c:
                            #   old_c becomes 0.
                            #   new_c becomes 1.
                            #   All others remain 1.
                            #   So the cycle remains valid!
                            #   Therefore, recoloring a valid cycle (with distinct colors) by changing one color to a *new* color not present in the cycle keeps it valid.
                            #   What if new_c IS present in the cycle?
                            #   If the cycle was valid, all colors are distinct. So new_c is not in the set of colors.
                            #   So new_c count is 0.
                            #   So recoloring always keeps a valid cycle valid?
                            #   Yes, if we change one color to a color not in the set, the set remains a set of 4 distinct colors.
                            
                            # So, recoloring an edge in a VALID cycle (present_count=4, distinct colors) to a color NOT in that cycle keeps it valid.
                            # If the color IS in that cycle, it becomes violated.
                            
                            # So, for a cycle c_idx containing e_idx:
                            # If c_idx is currently Valid (distinct):
                            #   If new_c is in the set of colors of c_idx, it becomes Violated.
                            #   Else, it remains Valid.
                            # If c_idx is currently Violated:
                            #   It might become Valid or remain Violated.
                            
                            # This logic is getting complex. Let's just re-evaluate is_violated for affected cycles.
                            # Since num_cycles per edge is small (~10-20), this is fast.
                            
                            # Simulate update
                            color_counts[c_idx][old_c] -= 1
                            color_counts[c_idx][new_c] += 1
                            if is_violated(c_idx):
                                new_violations += 1
                            # Revert
                            color_counts[c_idx][old_c] += 1
                            color_counts[c_idx][new_c] -= 1
                            
                    # Total delta for this move:
                    # -1 for the current cycle (fixed)
                    # + new_violations for other cycles
                    delta = new_violations - 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_move = ('recolor', e_idx, new_c)
                        
                    # If delta is negative, it's a great move.
                    if delta < 0:
                        break # Found a good recoloring for this edge
            
            if best_move and best_move[0] == 'recolor':
                _, e_idx, new_c = best_move
                old_c = color[e_idx]
                color[e_idx] = new_c
                
                # Update data structures
                for c_idx in edge_to_cycles[e_idx]:
                    color_counts[c_idx][old_c] -= 1
                    color_counts[c_idx][new_c] += 1
                    
                    if c_idx in violated_cycles:
                        if not is_violated(c_idx):
                            violated_cycles.remove(c_idx)
                    else:
                        if is_violated(c_idx):
                            violated_cycles.add(c_idx)
                fixed = True
                
            if not fixed:
                # Try to remove an edge
                # Pick an edge in the violated cycle that is in the most violated cycles
                # to maximize fix benefit.
                best_e_to_remove = None
                max_violations_fixed = -1
                for e_idx in edges_in_cyc:
                    count = 0
                    for c_idx in edge_to_cycles[e_idx]:
                        if c_idx in violated_cycles:
                            count += 1
                    if count > max_violations_fixed:
                        max_violations_fixed = count
                        best_e_to_remove = e_idx
                        
                if best_e_to_remove is not None:
                    e_idx = best_e_to_remove
                    present[e_idx] = 0
                    old_c = color[e_idx]
                    
                    for c_idx in edge_to_cycles[e_idx]:
                        present_count[c_idx] -= 1
                        color_counts[c_idx][old_c] -= 1
                        
                        if c_idx in violated_cycles:
                            # It was violated. Now present_count < 4, so not violated.
                            violated_cycles.remove(c_idx)
                        
            # Walk probability: make a random move
            if random.random() < walk_prob:
                # Random recolor or remove
                if random.random() < 0.5:
                    # Random recolor
                    e_idx = random.randint(0, m-1)
                    if present[e_idx]:
                        old_c = color[e_idx]
                        new_c = random.randint(0, num_colors-1)
                        if new_c != old_c:
                            color[e_idx] = new_c
                            for c_idx in edge_to_cycles[e_idx]:
                                color_counts[c_idx][old_c] -= 1
                                color_counts[c_idx][new_c] += 1
                                if c_idx in violated_cycles:
                                    if not is_violated(c_idx):
                                        violated_cycles.remove(c_idx)
                                else:
                                    if is_violated(c_idx):
                                        violated_cycles.add(c_idx)
                else:
                    # Random remove
                    present_edges = [i for i in range(m) if present[i]]
                    if present_edges:
                        e_idx = random.choice(present_edges)
                        present[e_idx] = 0
                        old_c = color[e_idx]
                        for c_idx in edge_to_cycles[e_idx]:
                            present_count[c_idx] -= 1
                            color_counts[c_idx][old_c] -= 1
                            if c_idx in violated_cycles:
                                violated_cycles.remove(c_idx)

        else:
            # No violations. Try to add an edge.
            absent_edges = [i for i in range(m) if not present[i]]
            if not absent_edges:
                break # Complete graph
            
            # Try to find an edge and color that doesn't create violations
            added = False
            # Shuffle absent edges
            random.shuffle(absent_edges)
            for e_idx in absent_edges:
                # Try all colors
                for c in range(num_colors):
                    # Check if adding e_idx with color c creates violations
                    new_violations = 0
                    for c_idx in edge_to_cycles[e_idx]:
                        if present_count[c_idx] == 3:
                            # Adding this edge makes it 4.
                            # Check if colors are distinct.
                            # Current colors of 3 edges + c.
                            # Distinct if c not in existing colors AND existing 3 are distinct.
                            
                            # Check existing 3 distinct
                            distinct = True
                            for cnt in color_counts[c_idx]:
                                if cnt > 1:
                                    distinct = False
                                    break
                            if not distinct:
                                new_violations += 1
                                continue
                                
                            if color_counts[c_idx][c] > 0:
                                new_violations += 1
                                continue
                            
                            # Else, no violation
                    if new_violations == 0:
                        # Apply move
                        present[e_idx] = 1
                        color[e_idx] = c
                        for c_idx in edge_to_cycles[e_idx]:
                            present_count[c_idx] += 1
                            color_counts[c_idx][c] += 1
                        added = True
                        break
                if added:
                    break
            
            if not added:
                # Stuck. Try to remove a random edge to escape local optimum?
                # Or just break?
                # Let's try to remove a random present edge and continue searching.
                present_edges = [i for i in range(m) if present[i]]
                if present_edges:
                    e_idx = random.choice(present_edges)
                    present[e_idx] = 0
                    old_c = color[e_idx]
                    for c_idx in edge_to_cycles[e_idx]:
                        present_count[c_idx] -= 1
                        color_counts[c_idx][old_c] -= 1
                        if c_idx in violated_cycles:
                            violated_cycles.remove(c_idx)
                else:
                    break

        # Update best
        current_edges = sum(present)
        if not violated_cycles and current_edges > best_edges:
            best_edges = current_edges
            best_present = present[:]
            best_color = color[:]

    # Extract result
    result = []
    for i in range(m):
        if best_present[i]:
            u, v = edge_list[i]
            c = best_color[i]
            result.append([u, v, c])
            
    return result

# EVOLVE-BLOCK-END
