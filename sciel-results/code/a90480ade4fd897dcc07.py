import random
import time
import itertools
import numpy as np
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
    cycles = []
    cycle_edges = [] # List of list of edge indices
    edge_to_cycles = defaultdict(list)
    
    vertices = range(n)
    for quad in itertools.combinations(vertices, 4):
        a, b, c, d = quad
        # 3 distinct cycles
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
            
            cyc_idx = tuple(sorted(cyc_idx_set))
            cycles.append(cyc_idx)
            cycle_edges.append(list(cyc_idx))
            
            for e_idx in cyc_idx:
                edge_to_cycles[e_idx].append(len(cycles) - 1)

    num_cycles = len(cycles)
    
    # State
    present = [1] * m
    color = [random.randint(0, num_colors - 1) for _ in range(m)]
    
    # Data structures
    present_count = [0] * num_cycles
    color_counts = [[0] * num_colors for _ in range(num_cycles)]
    
    # Initialize
    for i in range(m):
        if present[i]:
            c = color[i]
            for cyc_idx in edge_to_cycles[i]:
                present_count[cyc_idx] += 1
                color_counts[cyc_idx][c] += 1

    def is_violated(cyc_idx):
        if present_count[cyc_idx] < 4:
            return False
        counts = color_counts[cyc_idx]
        # Check if any count > 1
        for cnt in counts:
            if cnt > 1:
                return True
        return False

    violated_cycles = set()
    for i in range(num_cycles):
        if is_violated(i):
            violated_cycles.add(i)
            
    best_edges = sum(present)
    best_present = present[:]
    best_color = color[:]
    best_violations = len(violated_cycles)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Local Search
    while time.time() - start_time < time_limit:
        if violated_cycles:
            # Pick a random violated cycle
            cyc_idx = random.choice(list(violated_cycles))
            edges_in_cyc = cycle_edges[cyc_idx]
            
            # Analyze the violation
            # Get colors of the 4 edges
            cols = [color[e] for e in edges_in_cyc]
            
            # Try to fix by changing 1 edge color
            fixed = False
            # If there is a unique duplicate, we can fix it by changing one of the duplicates.
            # If there are two pairs, we need to change 2 edges or remove 1.
            
            # Count occurrences
            from collections import Counter
            cnt = Counter(cols)
            
            # Identify edges that are "part of a duplicate"
            dup_edges = [e for e in edges_in_cyc if cnt[color[e]] > 1]
            
            if len(dup_edges) == 2:
                # One pair. e.g., 1,1,2,3.
                # We can change one of the 1s to a new color not in {1,2,3}.
                # Or change the other 1.
                # Let's try changing one of the dup_edges.
                for e_idx in dup_edges:
                    old_c = color[e_idx]
                    other_cols = [color[e] for e in edges_in_cyc if e != e_idx]
                    # We need new_c not in other_cols
                    # other_cols has 3 elements.
                    # If other_cols has a duplicate, changing this edge won't fix the cycle
                    # because the other duplicate remains.
                    # But wait, if we have 1,1,2,3 and we change one 1 to 4, we get 4,1,2,3. Distinct!
                    # So we just need new_c not in {1,2,3}.
                    # The set of other colors is {1,2,3}.
                    # So we need new_c not in {1,2,3}.
                    # This is always possible since we have 20 colors.
                    
                    # However, we must ensure that changing this edge doesn't break other cycles.
                    # We will pick a random valid color and check the impact.
                    
                    available = [c for c in range(num_colors) if c not in other_cols]
                    if available:
                        new_c = random.choice(available)
                        
                        # Simulate and check if it reduces violations
                        # We'll just apply it and let the loop handle the consequences, 
                        # but we can be smarter: pick the color that minimizes new violations.
                        
                        # For speed, just pick one.
                        color[e_idx] = new_c
                        fixed = True
                        
                        # Update data structures
                        for c_idx in edge_to_cycles[e_idx]:
                            color_counts[c_idx][old_c] -= 1
                            color_counts[c_idx][new_c] += 1
                            
                        for c_idx in edge_to_cycles[e_idx]:
                            if c_idx in violated_cycles:
                                if not is_violated(c_idx):
                                    violated_cycles.remove(c_idx)
                            else:
                                if is_violated(c_idx):
                                    violated_cycles.add(c_idx)
                        break
                        
            elif len(dup_edges) == 4:
                # Two pairs or all same.
                # e.g., 1,1,2,2 or 1,1,1,1
                # Changing 1 edge won't fix it.
                # We must remove an edge or change 2 edges.
                # Let's try changing 2 edges.
                # Pick 2 edges to change.
                # This is complex. Let's just remove an edge for now.
                e_idx = random.choice(edges_in_cyc)
                present[e_idx] = 0
                old_c = color[e_idx]
                
                for c_idx in edge_to_cycles[e_idx]:
                    present_count[c_idx] -= 1
                    color_counts[c_idx][old_c] -= 1
                    
                    if c_idx in violated_cycles:
                        if not is_violated(c_idx):
                            violated_cycles.remove(c_idx)
                    else:
                        if is_violated(c_idx):
                            violated_cycles.add(c_idx)
                fixed = True
            else:
                # Should not happen if is_violated is true?
                # If is_violated, there must be a duplicate.
                # So dup_edges should be non-empty.
                pass
            
            if not fixed:
                 # Fallback: remove an edge
                 e_idx = random.choice(edges_in_cyc)
                 present[e_idx] = 0
                 old_c = color[e_idx]
                 
                 for c_idx in edge_to_cycles[e_idx]:
                     present_count[c_idx] -= 1
                     color_counts[c_idx][old_c] -= 1
                     
                     if c_idx in violated_cycles:
                         if not is_violated(c_idx):
                             violated_cycles.remove(c_idx)
                     else:
                         if is_violated(c_idx):
                             violated_cycles.add(c_idx)
        else:
            # No violations. Try to add an edge if we removed some.
            absent_edges = [i for i in range(m) if not present[i]]
            if not absent_edges:
                break # Max edges reached
            
            # Pick a random absent edge
            e_idx = random.choice(absent_edges)
            
            # Find a color that doesn't create violations
            best_c = -1
            min_violations = float('inf')
            
            affected = edge_to_cycles[e_idx]
            
            for c in range(num_colors):
                new_violations = 0
                for c_idx in affected:
                    if present_count[c_idx] == 3:
                        # Check if adding c creates a duplicate
                        # Check if c is already in color_counts[c_idx]
                        if color_counts[c_idx][c] > 0:
                            new_violations += 1
                        else:
                            # Check if the other 3 are distinct
                            # If they are not, adding c won't fix the existing duplicate, 
                            # so it will be a violation (4 edges, 3 distinct colors max? 
                            # No, if 3 are not distinct, they have a duplicate. 
                            # Adding a 4th edge makes 4 edges. 
                            # If the 3 had a duplicate, the 4 will have at least that duplicate.
                            # So it's a violation.
                            # We need to check if the 3 existing are distinct.
                            distinct = True
                            for cnt in color_counts[c_idx]:
                                if cnt > 1:
                                    distinct = False
                                    break
                            if not distinct:
                                new_violations += 1
                
                if new_violations < min_violations:
                    min_violations = new_violations
                    best_c = c
                    if min_violations == 0:
                        break
            
            # Apply best color
            color[e_idx] = best_c
            present[e_idx] = 1
            
            for c_idx in affected:
                present_count[c_idx] += 1
                color_counts[c_idx][best_c] += 1
                
                if c_idx in violated_cycles:
                    if not is_violated(c_idx):
                        violated_cycles.remove(c_idx)
                else:
                    if is_violated(c_idx):
                        violated_cycles.add(c_idx)
            
            # Update best
            current_edges = sum(present)
            current_violations = len(violated_cycles)
            # We prefer more edges, then fewer violations
            if (current_edges, -current_violations) > (best_edges, -best_violations):
                best_edges = current_edges
                best_violations = current_violations
                best_present = present[:]
                best_color = color[:]
        
        # Update best if we fixed violations with same edge count
        current_edges = sum(present)
        current_violations = len(violated_cycles)
        if (current_edges, -current_violations) > (best_edges, -best_violations):
            best_edges = current_edges
            best_violations = current_violations
            best_present = present[:]
            best_color = color[:]

    result = []
    for i in range(m):
        if best_present[i]:
            u, v = edge_list[i]
            c = best_color[i]
            result.append([u, v, c])
            
    return result
