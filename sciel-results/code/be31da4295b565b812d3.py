import time
import random
import sys
import itertools
import numpy as np

def construct():
    n = 20
    num_colors = 20
    deadline = time.time() + 115.0  # Safety margin
    
    # 1. Generate all 4-cycles
    # A 4-cycle is defined by 4 vertices. There are 3 distinct cycles for each set of 4 vertices.
    all_4cycles = []
    verts = range(n)
    for combo in itertools.combinations(verts, 4):
        a, b, c, d = combo
        # Cycle 1: a-b-c-d-a
        all_4cycles.append([(a,b), (b,c), (c,d), (d,a)])
        # Cycle 2: a-b-d-c-a
        all_4cycles.append([(a,b), (b,d), (d,c), (c,a)])
        # Cycle 3: a-c-b-d-a
        all_4cycles.append([(a,c), (c,b), (b,d), (d,a)])
        
    num_cycles = len(all_4cycles)
    
    # Pre-compute incidence: edge -> list of cycle indices
    edge_to_cycles = {}
    for i, cycle in enumerate(all_4cycles):
        for u, v in cycle:
            key = (min(u,v), max(u,v))
            if key not in edge_to_cycles:
                edge_to_cycles[key] = []
            edge_to_cycles[key].append(i)
            
    # Initialize graph with random colors
    # Using a 1-factorization as a start might be better, but random is fine if search is strong.
    # Let's try a structured start: color (u,v) with (u+v) % 20? No, that fails.
    # Let's use random.
    adj = [[-1] * n for _ in range(n)]
    edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for u, v in edges:
        c = random.randint(0, num_colors - 1)
        adj[u][v] = c
        adj[v][u] = c
        
    # State: current colors
    # We need to track the status of each cycle: is it rainbow?
    # cycle_status[i] = True if rainbow, False otherwise.
    # Also track count of bad cycles.
    
    def is_rainbow(cycle):
        colors = set()
        for u, v in cycle:
            colors.add(adj[u][v])
        return len(colors) == 4
        
    cycle_status = [is_rainbow(cycle) for cycle in all_4cycles]
    bad_count = sum(1 for s in cycle_status if not s)
    
    # Pre-compute for each cycle and each edge in it, the set of colors of the other 3 edges?
    # This depends on current colors, so it changes.
    # Instead, we will compute the delta on the fly for the affected cycles.
    
    best_bad = bad_count
    best_adj = [row[:] for row in adj]
    
    while time.time() < deadline:
        if best_bad == 0:
            break
            
        # Find a bad cycle
        # Random sampling might be faster than linear scan if bad_count is small, 
        # but linear scan is O(14k) which is fast.
        bad_cycle_idx = -1
        for i in range(num_cycles):
            if not cycle_status[i]:
                bad_cycle_idx = i
                break
                
        if bad_cycle_idx == -1:
            best_bad = 0
            best_adj = [row[:] for row in adj]
            break
            
        target_cycle = all_4cycles[bad_cycle_idx]
        
        # Try to fix this cycle by changing one of its edges
        improved = False
        # Shuffle edges in the cycle to avoid bias
        cycle_edges = target_cycle[:]
        random.shuffle(cycle_edges)
        
        for u, v in cycle_edges:
            key = (min(u,v), max(u,v))
            old_color = adj[u][v]
            affected_cycles = edge_to_cycles[key]
            
            # Try each new color
            # We can prioritize colors that are not in the other 3 edges of the target cycle
            # But we must check global impact.
            
            best_delta = 0
            best_new_color = -1
            
            # Optimization: Only try colors that might fix the target cycle?
            # If we pick a color that doesn't fix the target cycle, delta will likely be positive (worse) 
            # or zero. But it might fix other cycles? Unlikely if it doesn't fix the target one 
            # which is known to be bad.
            # Actually, if the target cycle is bad because of duplicates among the OTHER 3 edges, 
            # changing this edge won't fix it. We should detect this.
            
            other_colors = set()
            for eu, ev in target_cycle:
                if (min(eu,ev), max(eu,ev)) != key:
                    other_colors.add(adj[eu][ev])
                    
            if len(other_colors) < 3:
                # The other 3 edges have duplicates. Changing this edge cannot make this cycle rainbow.
                # Skip this edge for this target cycle? Or maybe try anyway to fix OTHER cycles?
                # If we don't fix the target cycle, we are not making progress on this specific bad cycle.
                # But we might reduce bad_count elsewhere? 
                # Let's just skip if we can't fix the target cycle, to focus search.
                continue
                
            # If len(other_colors) == 3, we need new_color not in other_colors.
            candidate_colors = [c for c in range(num_colors) if c not in other_colors]
            if not candidate_colors:
                continue
                
            random.shuffle(candidate_colors)
            
            for new_color in candidate_colors:
                if new_color == old_color:
                    continue
                    
                # Calculate delta in bad_count for all affected cycles
                delta = 0
                for ci in affected_cycles:
                    cyc = all_4cycles[ci]
                    was_bad = not cycle_status[ci]
                    
                    # Determine if it will be bad with new_color
                    # We need colors of the other 3 edges in this specific cycle
                    # Note: 'key' is the edge being changed.
                    o_c = set()
                    for eu, ev in cyc:
                        if (min(eu,ev), max(eu,ev)) != key:
                            o_c.add(adj[eu][ev])
                    
                    # Cycle is rainbow if new_color not in o_c and len(o_c) == 3
                    will_be_rainbow = (new_color not in o_c) and (len(o_c) == 3)
                    will_be_bad = not will_be_rainbow
                    
                    if was_bad and will_be_bad:
                        pass # stays bad
                    elif was_bad and not will_be_bad:
                        delta -= 1 # fixed
                    elif not was_bad and will_be_bad:
                        delta += 1 # broken
                    else:
                        pass # stays good
                        
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
                    
            if best_new_color != -1:
                # Apply the best change
                adj[u][v] = best_new_color
                adj[v][u] = best_new_color
                
                # Update cycle_status for affected cycles
                for ci in affected_cycles:
                    cycle_status[ci] = is_rainbow(all_4cycles[ci])
                    
                bad_count += best_delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_adj = [row[:] for row in adj]
                improved = True
                break # Move to next bad cycle
        
        if not improved:
            # Perturbation: Randomly change a random edge to a random color
            # and update status.
            u, v = random.choice(edges)
            key = (min(u,v), max(u,v))
            old_c = adj[u][v]
            new_c = random.randint(0, num_colors - 1)
            adj[u][v] = new_c
            adj[v][u] = new_c
            
            # Update affected cycles
            affected = edge_to_cycles[key]
            delta = 0
            for ci in affected:
                old_status = cycle_status[ci]
                new_status = is_rainbow(all_4cycles[ci])
                cycle_status[ci] = new_status
                if old_status and not new_status: delta += 1
                elif not old_status and new_status: delta -= 1
                
            bad_count += delta
            if bad_count < best_bad:
                best_bad = bad_count
                best_adj = [row[:] for row in adj]
                
            # If stuck, maybe restart with a different random seed?
            # For now, just keep perturbing.
            
    # Construct result from best_adj
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v] != -1:
                result_edges.append([u, v, best_adj[u][v]])
                
    return result_edges
