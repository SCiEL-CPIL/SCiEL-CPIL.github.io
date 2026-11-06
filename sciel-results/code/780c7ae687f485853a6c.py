import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices u, v, w, x.
    # The cycles are (u,v,w,x), (u,v,x,w), (u,w,v,x) in terms of edge sets.
    # Edges are stored as (min, max).
    
    all_cycles = []
    edge_to_cycles = {}
    
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 4):
        u, v, w, x = comb
        # Cycle 1: u-v-w-x-u
        c1 = [(u,v), (v,w), (w,x), (x,u)]
        # Cycle 2: u-v-x-w-u
        c2 = [(u,v), (v,x), (x,w), (w,u)]
        # Cycle 3: u-w-v-x-u
        c3 = [(u,w), (w,v), (v,x), (x,u)]
        
        for cycle in [c1, c2, c3]:
            # Normalize edges
            norm_cycle = []
            for e in cycle:
                norm_e = (min(e[0], e[1]), max(e[0], e[1]))
                norm_cycle.append(norm_e)
            # Check for duplicates in the cycle itself (shouldn't happen with distinct vertices)
            if len(set(norm_cycle)) == 4:
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    if e not in edge_to_cycles:
                        edge_to_cycles[e] = []
                    edge_to_cycles[e].append(len(all_cycles) - 1)

    # State
    # edge_colors: dict (u,v) -> color
    # valid_edges: set of (u,v)
    
    # Strategy:
    # 1. Start with a high-density random graph (p=0.9)
    # 2. Repair by removing edges that cause conflicts
    # 3. Local search: try to re-add removed edges or swap colors
    
    # Initialize with a dense random graph
    p = 0.9
    edge_colors = {}
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < p:
                edge_colors[(u,v)] = random.randint(0, num_colors - 1)
    
    def get_conflict_edges():
        """Return set of edges that are part of at least one non-rainbow 4-cycle."""
        conflict_edges = set()
        for cycle in all_cycles:
            # Check if all edges in cycle exist
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    # This cycle is not rainbow
                    # Mark all edges in this cycle as conflicting
                    for e in cycle:
                        conflict_edges.add(e)
        return conflict_edges
    
    def is_color_valid_for_edge(e, c, current_edge_colors):
        """Check if assigning color c to edge e results in all cycles containing e being rainbow."""
        if e not in edge_to_cycles:
            return True
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            # Check if all other edges in this cycle exist
            if not all(other_e in current_edge_colors for other_e in cycle if other_e != e):
                continue
            # Check colors
            other_colors = [current_edge_colors[other_e] for other_e in cycle if other_e != e]
            # The new color c must be distinct from all other_colors
            # AND the other_colors must be distinct from each other (otherwise cycle is already broken)
            if c in other_colors:
                return False
            # If other colors have duplicates, the cycle is already invalid, but adding e with any color won't fix it.
            # However, if we are repairing, we assume we are trying to make it valid.
            # If the other 3 are not distinct, this cycle is fundamentally broken unless we change those edges too.
            # For a single-edge move to be "valid" in the sense of not *creating* a new conflict from a valid state:
            # If state was valid, other 3 are distinct. Then c must not be in them.
            # If state is invalid, this check is less meaningful for "repairing" a single edge.
            # But for "adding an edge to a valid state", this is correct.
            # Let's assume we are maintaining or moving towards a valid state.
            # If other_colors are not distinct, the cycle is already bad.
            # If we add e, it remains bad.
            # So does this move "break" anything? No, it was already broken.
            # But does it "fix" anything? Only if the badness was caused by e's absence? No, e was absent.
            # Actually, if e is absent, the cycle is not a 4-cycle in the graph.
            # The condition is "Every 4-cycle (as a subgraph) must receive four distinct colours".
            # If an edge is missing, it's not a 4-cycle.
            # So adding e creates a 4-cycle.
            # If the other 3 edges have colors [1, 1, 2], the new 4-cycle has colors [c, 1, 1, 2].
            # This is NOT rainbow.
            # So if the other 3 are not distinct, we CANNOT add e with any color and have a valid 4-cycle.
            # Thus, is_color_valid_for_edge should return False if other_colors are not distinct.
            
            if len(set(other_colors)) < 3:
                return False
            
            # If other_colors are distinct, c must not be in them.
            # We already checked `if c in other_colors: return False`
            
        return True

    # Phase 1: Repair
    # While there are conflicts, remove an edge from a conflict cycle.
    # Heuristic: Remove the edge that is in the most conflict cycles?
    # Or just random removal from conflict set.
    
    max_repair_steps = 500
    for _ in range(max_repair_steps):
        if time.time() > deadline:
            break
            
        conflict_edges = get_conflict_edges()
        if not conflict_edges:
            break
        
        # Pick a random conflict edge to remove
        # Or pick the one with highest degree in conflict graph?
        # Let's stick to random for now, but maybe weighted by frequency?
        # Simple random is fast.
        e_to_remove = random.choice(list(conflict_edges))
        del edge_colors[e_to_remove]
        
    # Phase 2: Local Search
    # Try to add back removed edges or swap colors to allow more edges.
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    current_edges = set(edge_colors.keys())
    missing_edges = [e for e in all_edges if e not in current_edges]
    
    # We want to maximize edges.
    # Try to add a missing edge.
    # If we can't, try to recolor existing edges to make room?
    # Recoloring is complex.
    # Let's try:
    # 1. Pick a missing edge e.
    # 2. Check if there's a valid color.
    # 3. If not, identify why (which cycles block it).
    # 4. Maybe remove one edge from a blocking cycle and re-add e?
    
    # Simpler Local Search:
    # 1. Pick a random missing edge e.
    # 2. Try to add it with a random valid color.
    # 3. If successful, great.
    # 4. If not, pick a random existing edge f that shares a 4-cycle with e.
    #    Remove f, try to add e.
    #    If e is added, we have same number of edges.
    #    Then try to add f back?
    #    This is a swap.
    
    # Even better:
    # 1. Pick a random missing edge e.
    # 2. Try to add it.
    # 3. If fail, try to find a set of edges to remove (size 1) such that e can be added, 
    #    and then the removed edge can be re-added? No, that's 0 gain.
    #    We want +1 edge.
    #    So we need to add e without removing anything.
    #    If we can't, we are stuck locally.
    #    To escape, we might need to remove 2 edges and add 3? Too complex.
    
    # Let's try a different local search:
    # Randomly recolor an existing edge.
    # This might resolve conflicts and allow adding a missing edge.
    
    attempts = 0
    best_edge_count = len(edge_colors)
    
    while time.time() < deadline:
        attempts += 1
        
        # 1. Try to add a missing edge
        if missing_edges:
            e_add = random.choice(missing_edges)
            colors = list(range(num_colors))
            random.shuffle(colors)
            added = False
            for c in colors:
                if is_color_valid_for_edge(e_add, c, edge_colors):
                    edge_colors[e_add] = c
                    missing_edges.remove(e_add)
                    added = True
                    if len(edge_colors) > best_edge_count:
                        best_edge_count = len(edge_colors)
                    break
            
            if added:
                continue
        
        # 2. If couldn't add, try a random recolor
        if edge_colors:
            e_recolor = random.choice(list(edge_colors.keys()))
            old_color = edge_colors[e_recolor]
            
            # Try a new random color
            new_color = random.randint(0, num_colors - 1)
            if new_color != old_color:
                # Temporarily remove
                del edge_colors[e_recolor]
                
                # Check if new color is valid
                if is_color_valid_for_edge(e_recolor, new_color, edge_colors):
                    edge_colors[e_recolor] = new_color
                else:
                    # Revert
                    edge_colors[e_recolor] = old_color
        
        # 3. Occasionally, try a "swap" move
        # Remove a random edge, add a missing one, then try to re-add the removed one?
        # This is essentially a 2-opt move.
        
        if random.random() < 0.1 and edge_colors and missing_edges:
            # Pick a random existing edge to remove
            e_remove = random.choice(list(edge_colors.keys()))
            del edge_colors[e_remove]
            
            # Pick a missing edge to add
            e_add = random.choice(missing_edges)
            colors = list(range(num_colors))
            random.shuffle(colors)
            added = False
            for c in colors:
                if is_color_valid_for_edge(e_add, c, edge_colors):
                    edge_colors[e_add] = c
                    missing_edges.remove(e_add)
                    added = True
                    break
            
            if added:
                # Now try to re-add e_remove
                colors = list(range(num_colors))
                random.shuffle(colors)
                for c in colors:
                    if is_color_valid_for_edge(e_remove, c, edge_colors):
                        edge_colors[e_remove] = c
                        # If we re-added it, net change is 0 (removed 1, added 1, re-added 1 -> +1? No.)
                        # Removed e_remove. Added e_add. Re-added e_remove.
                        # Net: +1 (e_add is new).
                        if len(edge_colors) > best_edge_count:
                            best_edge_count = len(edge_colors)
                        break
            else:
                # e_add couldn't be added. Re-add e_remove.
                # This was a wasted move.
                colors = list(range(num_colors))
                random.shuffle(colors)
                for c in colors:
                    if is_color_valid_for_edge(e_remove, c, edge_colors):
                        edge_colors[e_remove] = c
                        break
                # If we couldn't re-add e_remove either, we lost an edge!
                # We should avoid this.
                # Let's check if e_remove is in edge_colors. If not, it's gone.
                # This is bad for maximizing edges.
                # We should only do swaps if we are confident.
                # For now, if we lost an edge, we just continue.
                # But we want to maximize. So we should revert if we lose an edge.
                # How to revert? We don't have a stack.
                # So we should be careful.
                # Maybe only do swaps if we can guarantee re-adding?
                # Or just skip this part if it's risky.
                
                # Let's check if we actually lost an edge.
                # If e_remove is not in edge_colors, we lost it.
                # If e_add was not added, and e_remove is not re-added, we are at -1 edge.
                # This is bad.
                # We should probably avoid this heuristic or make it safer.
                # Let's just skip the swap if it's not clearly beneficial.
                # Actually, if e_add was NOT added, we should have just re-added e_remove immediately.
                # The code above does that.
                # But if re-adding e_remove FAILS, we are stuck with -1 edge.
                # This can happen if the removal of e_remove changed the validity of e_remove itself?
                # No, e_remove was valid before. Removing it makes it absent.
                # Adding it back with the same color should be valid?
                # Wait, is_color_valid_for_edge checks against CURRENT edge_colors.
                # When we removed e_remove, the cycles containing it lost one edge.
                # When we try to add it back, the other 3 edges are still there (assuming they weren't removed).
                # So the check should pass with the old color.
                # So re-adding with old color should always work?
                # Yes, because the state of other edges didn't change (except e_add was added).
                # If e_add was added, it might conflict with e_remove?
                # If e_add and e_remove share a 4-cycle, adding e_add might make e_remove invalid.
                # So re-adding e_remove might fail.
                # In that case, we have +1 (e_add) and -1 (e_remove) = 0 net change?
                # No, we removed e_remove, added e_add.
                # If we can't re-add e_remove, we have net 0 change (lost e_remove, gained e_add).
                # Wait, e_add was missing, so gaining it is +1.
                # Losing e_remove is -1.
                # Net 0.
                # So we didn't lose an edge count, we just swapped.
                # But if e_add was NOT added, we just removed e_remove and failed to add e_add.
                # Then we try to re-add e_remove.
                # If re-add fails, we lost e_remove. Net -1.
                # This is the bad case.
                # To avoid this, if e_add fails, we MUST re-add e_remove.
                # And re-adding e_remove with old color should work because e_add wasn't added.
                # So the state is identical to before the swap.
                # So re-adding e_remove should always succeed if e_add failed.
                
                # So the only risk is if e_add succeeds, but e_remove fails to re-add.
                # In that case, we have net 0 change (swapped e_remove for e_add).
                # This is acceptable.
                
                # So the heuristic is safe in terms of not decreasing edge count?
                # Case 1: e_add added, e_remove re-added. Net +1.
                # Case 2: e_add added, e_remove NOT re-added. Net 0 (swap).
                # Case 3: e_add NOT added, e_remove re-added. Net 0 (no-op).
                # Case 4: e_add NOT added, e_remove NOT re-added. Net -1. (Should not happen).
                
                # So it's safe.
                
                pass
    
    # Final check
    # Ensure no conflicts
    conflict_edges = get_conflict_edges()
    while conflict_edges:
        if time.time() > deadline:
            break
        e_to_remove = random.choice(list(conflict_edges))
        del edge_colors[e_to_remove]
        conflict_edges = get_conflict_edges()
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
        
    return result
