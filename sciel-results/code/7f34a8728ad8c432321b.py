import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    n = 20
    colors_count = 20
    deadline = time.time() + 110.0  # Safety margin

    # Precompute edges
    edges = []
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[(u, v)] = idx
            edges.append((u, v))
            idx += 1
    num_edges = len(edges)

    # Precompute 4-cycles
    # A 4-cycle is represented by a tuple of 4 edge indices.
    # We generate all sets of 4 vertices, and for each, the 3 distinct 4-cycles.
    cycles = []
    for i in range(n-3):
        for j in range(i+1, n-2):
            for k in range(j+1, n-1):
                for l in range(k+1, n):
                    u, v, w, x = i, j, k, l
                    # 3 cycles on {u, v, w, x}
                    # 1. u-v-w-x-u
                    e1 = edge_map.get((min(u,v), max(u,v)))
                    e2 = edge_map.get((min(v,w), max(v,w)))
                    e3 = edge_map.get((min(w,x), max(w,x)))
                    e4 = edge_map.get((min(x,u), max(x,u)))
                    cycles.append((e1, e2, e3, e4))
                    
                    # 2. u-v-x-w-u
                    e1 = edge_map.get((min(u,v), max(u,v)))
                    e2 = edge_map.get((min(v,x), max(v,x)))
                    e3 = edge_map.get((min(x,w), max(x,w)))
                    e4 = edge_map.get((min(w,u), max(w,u)))
                    cycles.append((e1, e2, e3, e4))
                    
                    # 3. u-w-v-x-u
                    e1 = edge_map.get((min(u,w), max(u,w)))
                    e2 = edge_map.get((min(w,v), max(v,v))) # Wait, min(w,v) is fine
                    e2 = edge_map.get((min(w,v), max(w,v)))
                    e3 = edge_map.get((min(v,x), max(v,x)))
                    e4 = edge_map.get((min(x,u), max(x,u)))
                    cycles.append((e1, e2, e3, e4))
    
    num_cycles = len(cycles)

    # Map edge index to list of cycle indices it participates in
    edge_to_cycles = [[] for _ in range(num_edges)]
    for c_idx, cycle in enumerate(cycles):
        for e_idx in cycle:
            edge_to_cycles[e_idx].append(c_idx)
            
    # For each edge, precompute the list of cycles where it is the "newest" or simply just
    # the cycles it belongs to. We need to check if adding an edge creates a violation.
    # A violation exists if a cycle has all 4 edges present and colors are not distinct.
    
    # Strategy: Greedy Edge Addition
    # 1. Start with an empty graph (no edges).
    # 2. Consider edges in a specific order (randomized for restarts).
    # 3. For each edge, try to assign a color such that no 4-cycle containing this edge
    #    and 3 other existing edges becomes a violation.
    #    Note: If the 3 other edges don't exist, no violation is possible.
    #    If they exist, we must pick a color different from the other 3.
    
    best_result = None
    num_restarts = 100
    
    # Helper to check if a coloring is valid for a given set of active edges
    # Actually, we maintain validity incrementally.
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Randomize edge order
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        # active_edges: boolean array
        active_edges = [False] * num_edges
        # colors: array of colors for active edges
        colors = [0] * num_edges
        
        # To speed up checking, we can maintain a count of how many edges in each cycle are active.
        # cycle_edge_count[c_idx] = number of active edges in cycle c_idx
        cycle_edge_count = [0] * num_cycles
        
        # When we add an edge, we increment counts for cycles it belongs to.
        # If a count reaches 4, we must check if the 4 colors are distinct.
        # If not distinct, we cannot add this edge with this color, or we must choose a different color.
        
        # However, if a cycle already has 3 active edges, and we add the 4th, we MUST ensure
        # the new color is different from the other 3.
        # If a cycle has < 3 active edges, adding a 4th is impossible (we are adding 1, so if it was 3, it becomes 4).
        # Wait, if a cycle has 2 active edges, adding 1 makes it 3. No violation possible yet.
        # If a cycle has 3 active edges, adding 1 makes it 4. Violation possible.
        
        # So, for a candidate edge e:
        # 1. Identify all cycles C containing e.
        # 2. For each C, if cycle_edge_count[C] == 3:
        #    - The other 3 edges in C are active.
        #    - Let their colors be c1, c2, c3.
        #    - The new color c must be != c1, != c2, != c3.
        # 3. If cycle_edge_count[C] < 3, no constraint from C (since we can't form a 4-cycle with fewer than 3 existing edges).
        # 4. If cycle_edge_count[C] > 3, that's impossible because we only add edges one by one and start from 0.
        #    Actually, can a cycle have > 3 active edges? No, max is 4.
        
        # So the constraint is: For every cycle C containing e where 3 other edges are already active,
        # the chosen color for e must not be in {colors of those 3 edges}.
        
        # If there are no such constraints, any color works.
        # If there are constraints, we pick a color from the allowed set (20 - number of forbidden colors).
        
        # If the allowed set is empty, we cannot add this edge.
        
        # Let's implement this.
        
        # To make it faster, we can precompute for each edge e, the list of cycles it is in.
        # We already have edge_to_cycles.
        
        # We need to quickly find the other 3 edges in a cycle if 3 are active.
        # Since we maintain cycle_edge_count, we know if it's 3.
        # We need to find which 3 are active.
        # We can just iterate the 4 edges in the cycle and check active_edges.
        
        current_edge_count = 0
        
        # Optimization: If we fail to add an edge, we skip it.
        # We might want to retry with different randomizations or different edge orders.
        
        for e_idx in edge_order:
            if time.time() > deadline - 2.0:
                break
                
            # Determine forbidden colors
            forbidden = set()
            can_add = True
            
            for c_idx in edge_to_cycles[e_idx]:
                if cycle_edge_count[c_idx] == 3:
                    # Find the 3 active edges in this cycle
                    c1, c2, c3, c4 = cycles[c_idx]
                    active_colors = []
                    # We know e_idx is one of these, but not active yet.
                    # The other 3 are active.
                    for e_other in (c1, c2, c3, c4):
                        if e_other != e_idx and active_edges[e_other]:
                            active_colors.append(colors[e_other])
                    # There should be exactly 3 such colors
                    if len(active_colors) == 3:
                        forbidden.update(active_colors)
                    else:
                        # Something went wrong, but logically this shouldn't happen if counts are correct
                        pass
                elif cycle_edge_count[c_idx] > 3:
                    # Should not happen
                    can_add = False
                    break
            
            if not can_add:
                continue
            
            # Pick a color
            if len(forbidden) < colors_count:
                # Choose a random color not in forbidden
                available = [c for c in range(colors_count) if c not in forbidden]
                new_color = random.choice(available)
                
                # Add the edge
                active_edges[e_idx] = True
                colors[e_idx] = new_color
                current_edge_count += 1
                
                # Update cycle counts
                for c_idx in edge_to_cycles[e_idx]:
                    cycle_edge_count[c_idx] += 1
            else:
                # No available color
                continue
        
        # End of edge order loop
        # Check if this result is better than best
        if best_result is None or current_edge_count > len(best_result):
            # Construct the result list
            result = []
            for i in range(num_edges):
                if active_edges[i]:
                    u, v = edges[i]
                    result.append([u, v, colors[i]])
            best_result = result
            
            # Optional: Local search improvement on the valid graph?
            # Since the graph is already valid, we can try to add more edges?
            # No, we exhausted the edge order.
            # But maybe a different order would allow more edges?
            # The restarts handle that.
            
            # Can we try to fill in more edges that were skipped?
            # We can do a second pass: for skipped edges, check if they can be added now.
            # But the state is fixed. If an edge couldn't be added before, it might still be blocked.
            # Actually, adding edges only increases constraints. It never relaxes them.
            # So if an edge was blocked by 3 existing edges, adding more edges won't help unblock it.
            # Wait, is that true?
            # Constraint: New color must differ from 3 others.
            # If I add more edges, I don't change the colors of the existing 3.
            # So the forbidden set for e remains the same (or grows if new cycles become full? No, cycles are fixed).
            # Actually, a cycle becomes "full" (4 edges) only when the 4th is added.
            # If a cycle has 3 edges, and I add a 4th, it becomes 4.
            # If I skip e, and later add other edges, does that affect e's ability to be added?
            # No, because e's constraints come from cycles where the OTHER 3 are already present.
            # Adding more edges elsewhere doesn't change the presence of those 3.
            # So, if e is blocked, it stays blocked.
            # Therefore, the greedy pass is final for that ordering.
            
            # However, if we use local search to RECOLOR existing edges, we might free up a color for e?
            # Yes! Recoloring an edge in a cycle reduces the "fixed" nature of the forbidden set?
            # No, the forbidden set is determined by the CURRENT colors of the 3 active edges.
            # If we recolor one of those 3 edges, the forbidden set for e changes.
            # So, we could try to recolor existing edges to allow adding skipped ones.
            # This is complex. Let's stick to multiple restarts with different orders.
            
            # If we found a very dense graph (e.g., > 150), we might stop early?
            # Let's just run all restarts within time.

    # Fallback if no edges were added (should not happen, star is always possible)
    if best_result is None:
        return [[0, v, 0] for v in range(1, 20)]
        
    return best_result

# EVOLVE-BLOCK-END
