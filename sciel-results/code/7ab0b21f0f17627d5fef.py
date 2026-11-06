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
    cycle_indices = [] # List of list of edge indices
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
            # Normalize and get indices
            cyc_idx_set = []
            for e in edges:
                u, v = e
                if u > v:
                    u, v = v, u
                cyc_idx_set.append(edge_to_idx[(u, v)])
            
            cyc_idx = tuple(sorted(cyc_idx_set))
            cycles.append(cyc_idx)
            cycle_indices.append(cyc_idx)
            
            for e_idx in cyc_idx:
                edge_to_cycles[e_idx].append(len(cycles) - 1)

    num_cycles = len(cycles)
    
    # State variables
    present = [0] * m
    color = [0] * m
    
    # Data structures for fast violation tracking
    present_count = [0] * num_cycles
    color_counts = [[0] * num_colors for _ in range(num_cycles)]
    
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
    
    def update_violation_status(c_idx):
        if c_idx in violated_cycles:
            if not is_violated(c_idx):
                violated_cycles.remove(c_idx)
        else:
            if is_violated(c_idx):
                violated_cycles.add(c_idx)

    def get_score():
        edges = sum(present)
        return edges - 1000 * len(violated_cycles)

    # Initialize with empty graph
    present = [0] * m
    color = [0] * m
    present_count = [0] * num_cycles
    color_counts = [[0] * num_colors for _ in range(num_cycles)]
    violated_cycles = set()
    
    best_present = present[:]
    best_color = color[:]
    best_score = 0
    
    start_time = time.time()
    time_limit = 105.0
    
    # Walk parameters
    walk_prob = 0.3
    max_walk_steps = 10
    
    while time.time() - start_time < time_limit:
        # If there are violations, try to fix them
        if violated_cycles:
            # Pick a random violated cycle
            cyc_idx = random.choice(list(violated_cycles))
            edges_in_cyc = cycle_indices[cyc_idx]
            
            # Strategy: Try to fix by changing color or removing edge
            # We evaluate all possible moves (change color of one of 4 edges, or remove one of 4 edges)
            # and pick the one that minimizes the number of violations (or maximizes score).
            
            best_move_score = -1
            best_move = None
            
            # Option A: Change color
            for e_idx in edges_in_cyc:
                old_color = color[e_idx]
                other_colors = set()
                for other_e in edges_in_cyc:
                    if other_e != e_idx:
                        other_colors.add(color[other_e])
                
                # Only consider colors not in other_colors if others are distinct
                if len(other_colors) < 3:
                    continue # Can't fix by changing this edge if others are duplicated
                
                for new_color in range(num_colors):
                    if new_color in other_colors:
                        continue
                    if new_color == old_color:
                        continue
                    
                    # Simulate move
                    # Update color_counts for affected cycles
                    affected = edge_to_cycles[e_idx]
                    delta_violations = 0
                    for c_idx in affected:
                        if c_idx == cyc_idx:
                            # We know this one gets fixed (since we chose a valid color)
                            delta_violations -= 1
                            continue
                        
                        old_state = is_violated(c_idx)
                        
                        # Apply change
                        color_counts[c_idx][old_color] -= 1
                        color_counts[c_idx][new_color] += 1
                        
                        new_state = is_violated(c_idx)
                        
                        if new_state and not old_state:
                            delta_violations += 1
                        elif old_state and not new_state:
                            delta_violations -= 1
                        
                        # Revert
                        color_counts[c_idx][new_color] -= 1
                        color_counts[c_idx][old_color] += 1
                    
                    score_change = -delta_violations * 1000
                    if score_change > best_move_score:
                        best_move_score = score_change
                        best_move = ('color', e_idx, new_color)
            
            # Option B: Remove edge
            for e_idx in edges_in_cyc:
                old_color = color[e_idx]
                affected = edge_to_cycles[e_idx]
                delta_violations = 0
                for c_idx in affected:
                    if c_idx == cyc_idx:
                        delta_violations -= 1
                        continue
                    
                    old_state = is_violated(c_idx)
                    
                    # Apply removal
                    present_count[c_idx] -= 1
                    color_counts[c_idx][old_color] -= 1
                    
                    new_state = is_violated(c_idx)
                    
                    if new_state and not old_state:
                        delta_violations += 1
                    elif old_state and not new_state:
                        delta_violations -= 1
                    
                    # Revert
                    present_count[c_idx] += 1
                    color_counts[c_idx][old_color] += 1
                
                score_change = -1 - delta_violations * 1000
                if score_change > best_move_score:
                    best_move_score = score_change
                    best_move = ('remove', e_idx, None)
            
            # Apply best move
            if best_move:
                move_type, e_idx, new_color = best_move
                if move_type == 'color':
                    old_color = color[e_idx]
                    color[e_idx] = new_color
                    for c_idx in edge_to_cycles[e_idx]:
                        color_counts[c_idx][old_color] -= 1
                        color_counts[c_idx][new_color] += 1
                        update_violation_status(c_idx)
                else: # remove
                    old_color = color[e_idx]
                    present[e_idx] = 0
                    for c_idx in edge_to_cycles[e_idx]:
                        present_count[c_idx] -= 1
                        color_counts[c_idx][old_color] -= 1
                        update_violation_status(c_idx)
            
        else:
            # No violations. Try to add an edge.
            absent_edges = [i for i in range(m) if not present[i]]
            if not absent_edges:
                break
                
            # Pick a random edge
            e_idx = random.choice(absent_edges)
            
            # Find best color
            best_c = -1
            best_score_change = -1
            affected = edge_to_cycles[e_idx]
            
            for c in range(num_colors):
                delta_violations = 0
                for c_idx in affected:
                    if present_count[c_idx] == 3:
                        # Check if adding edge with color c creates violation
                        # Existing 3 must be distinct and c not in them
                        existing_distinct = True
                        for cnt in color_counts[c_idx]:
                            if cnt > 1:
                                existing_distinct = False
                                break
                        if not existing_distinct:
                            delta_violations += 1
                            continue
                        if color_counts[c_idx][c] > 0:
                            delta_violations += 1
                
                score_change = 1 - delta_violations * 1000
                if score_change > best_score_change:
                    best_score_change = score_change
                    best_c = c
            
            # If best move is negative, we might still do it to escape local optima?
            # Or skip. Let's skip if it's bad.
            if best_score_change < 0:
                continue
            
            # Apply
            color[e_idx] = best_c
            present[e_idx] = 1
            for c_idx in affected:
                present_count[c_idx] += 1
                color_counts[c_idx][best_c] += 1
                update_violation_status(c_idx)
            
        # Update best
        current_score = get_score()
        if current_score > best_score:
            best_score = current_score
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
