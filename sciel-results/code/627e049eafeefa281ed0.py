import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n = 20
    n_colors = 20
    
    # 1. Generate all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    num_edges = len(all_edges)

    # 2. Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices. There are 3 such cycles per set of 4 vertices.
    all_4cycles = []
    for quad in combinations(range(n), 4):
        w, x, y, z = quad
        cycles = [
            [(w, x), (x, y), (y, z), (z, w)],
            [(w, x), (x, z), (z, y), (y, w)],
            [(w, y), (y, x), (x, z), (z, w)],
        ]
        for cyc in cycles:
            # Convert to edge indices
            try:
                edge_idxs = tuple(sorted(edge_to_idx[e] for e in cyc))
                all_4cycles.append(edge_idxs)
            except KeyError:
                pass

    # Map edge index to list of 4-cycles containing it
    edge_cycles = [[] for _ in range(num_edges)]
    for ci, cyc in enumerate(all_4cycles):
        for ei in cyc:
            edge_cycles[ei].append(ci)

    # 3. Initialize with a proper edge coloring (1-factorization of K20)
    # K20 has degree 19, so it can be decomposed into 19 perfect matchings.
    # We assign colors 0..18 to these matchings.
    # We have 1 extra color (19) available.
    
    # Simple 1-factorization algorithm for K_{2n} (here 2n=20, n=10)
    # Vertices 0..18 are arranged in a circle, 19 is the center.
    # Actually, standard construction for K_{2n}:
    # Vertices 0..2n-2 in a circle, 2n-1 is center.
    # For k in 0..n-1:
    #   Matching M_k consists of edges (center, 2n-1) is not right.
    # Let's use a simpler explicit construction for K_20.
    # Or just use a greedy proper coloring.
    
    colors = [-1] * num_edges
    active_edges = [True] * num_edges
    
    # Greedy proper edge coloring
    # Order edges by degree of endpoints? Or just random.
    # To get a 1-factorization, we can use the standard algorithm.
    # But a random proper coloring might be easier to implement and sufficient if we have 20 colors.
    # Chromatic index of K20 is 19. We have 20. So a proper coloring exists.
    
    random.shuffle(all_edges) # Shuffle for randomization
    edge_order = list(range(num_edges))
    random.shuffle(edge_order)
    
    for ei in edge_order:
        u, v = all_edges[ei]
        # Find forbidden colors based on neighbors
        forbidden = set()
        # Check all edges incident to u or v
        for (a, b) in all_edges:
            if a == u or a == v or b == u or b == v:
                idx = edge_to_idx[(min(a,b), max(a,b))]
                if colors[idx] != -1:
                    forbidden.add(colors[idx])
        
        # Assign first available color
        for c in range(n_colors):
            if c not in forbidden:
                colors[ei] = c
                break
        else:
            # Should not happen with 20 colors for K20
            colors[ei] = 0 

    # 4. Local Search to fix 4-cycle rainbow violations
    # A 4-cycle is invalid if its 4 edge colors are not distinct.
    # Since we started with a proper coloring, adjacent edges have different colors.
    # So the only violation is if opposite edges have the same color.
    
    def is_valid(cyc_idx):
        cyc = all_4cycles[cyc_idx]
        if not all(active_edges[e] for e in cyc):
            return True
        c1 = colors[cyc[0]]
        c2 = colors[cyc[1]]
        c3 = colors[cyc[2]]
        c4 = colors[cyc[3]]
        return len({c1, c2, c3, c4}) == 4

    # Track number of invalid cycles
    num_invalid = 0
    invalid_cycles = set()
    
    # Initial count
    for ci in range(len(all_4cycles)):
        if not is_valid(ci):
            num_invalid += 1
            invalid_cycles.add(ci)

    best_num_edges = sum(active_edges)
    best_colors = colors[:]
    best_active = active_edges[:]

    def try_recolor(ei, new_color):
        nonlocal num_invalid, invalid_cycles
        old_color = colors[ei]
        if old_color == new_color:
            return 0
        
        # Check impact on all 4-cycles containing ei
        delta = 0
        affected_cycles = edge_cycles[ei]
        
        # Tentatively change
        colors[ei] = new_color
        
        # Update validity of affected cycles
        for ci in affected_cycles:
            was_valid = is_valid(ci)
            # Note: is_valid uses global colors, so it sees the new color
            is_now_valid = is_valid(ci)
            
            if was_valid and not is_now_valid:
                delta -= 1
                invalid_cycles.add(ci)
            elif not was_valid and is_now_valid:
                delta += 1
                invalid_cycles.remove(ci)
            elif not was_valid and not is_now_valid:
                # Still invalid, no change in count, but set membership might need update?
                # Actually, if it was invalid and is still invalid, the set doesn't change.
                pass
            elif was_valid and is_now_valid:
                pass

        num_invalid += delta
        
        # Revert if not better
        if delta < 0:
            colors[ei] = old_color
            # Revert set changes
            for ci in affected_cycles:
                was_valid = is_valid(ci) # This is now with old color
                # This logic is getting messy. Let's just recalculate affected cycles' status properly.
                pass
            # Better approach: just keep track of a list of (cycle, old_valid, new_valid)
            return delta # Return negative to indicate bad move, caller reverts
            
        return delta

    # Simpler local search loop
    for iteration in range(5000): # Max iterations
        if time.time() > deadline - 5.0:
            break
        
        if num_invalid == 0:
            break
            
        # Pick a random invalid cycle
        if not invalid_cycles:
            break
        ci = random.choice(list(invalid_cycles))
        cyc = all_4cycles[ci]
        
        # Pick a random edge in this cycle to recolor
        ei = random.choice(cyc)
        
        # Try to find a better color for ei
        best_delta = 0
        best_c = colors[ei]
        
        # Candidate colors: current color and others
        # We can try all 20 colors
        for c in range(n_colors):
            if c == colors[ei]:
                continue
            
            # Calculate delta for changing ei to c
            old_color = colors[ei]
            colors[ei] = c
            
            delta = 0
            for aci in edge_cycles[ei]:
                # Check if this cycle is in the current graph
                if not all(active_edges[e] for e in all_4cycles[aci]):
                    continue
                    
                # Check validity before and after?
                # We changed colors[ei].
                # We need to know if aci was valid BEFORE and if it is valid NOW.
                # Since we only changed one edge, only cycles containing ei are affected.
                
                # To avoid expensive full check, we can just check the 4 colors of aci.
                # But we need the "before" state.
                # Let's just re-evaluate validity of affected cycles.
                # Optimization: Pre-compute validity? No, colors change.
                
                # Simple check:
                # Get colors of the 4 edges in aci
                c0 = colors[all_4cycles[aci][0]]
                c1 = colors[all_4cycles[aci][1]]
                c2 = colors[all_4cycles[aci][2]]
                c3 = colors[all_4cycles[aci][3]]
                is_valid_now = len({c0, c1, c2, c3}) == 4
                
                # To know "before", we need to revert colors[ei] temporarily?
                # Or store the old color.
                # Let's just revert colors[ei] to old_color for the check, then set to c.
                # This is slow.
                
                # Alternative:
                # If aci was in invalid_cycles, it was invalid.
                # If aci was NOT in invalid_cycles, it was valid.
                # BUT, invalid_cycles might not be perfectly synced if we don't update it carefully.
                # Let's assume invalid_cycles is correct.
                
                was_valid = aci not in invalid_cycles
                # Note: is_valid_now is calculated with new color c.
                
                if was_valid and not is_valid_now:
                    delta -= 1
                elif not was_valid and is_valid_now:
                    delta += 1
                
            # Revert color to old_color for next candidate
            colors[ei] = old_color
            
            if delta > best_delta:
                best_delta = delta
                best_c = c
        
        # Apply best color if it improves
        if best_delta > 0:
            colors[ei] = best_c
            # Update invalid_cycles
            for aci in edge_cycles[ei]:
                if not all(active_edges[e] for e in all_4cycles[aci]):
                    continue
                c0 = colors[all_4cycles[aci][0]]
                c1 = colors[all_4cycles[aci][1]]
                c2 = colors[all_4cycles[aci][2]]
                c3 = colors[all_4cycles[aci][3]]
                is_valid_now = len({c0, c1, c2, c3}) == 4
                
                if is_valid_now:
                    if aci in invalid_cycles:
                        invalid_cycles.remove(aci)
                        num_invalid -= 1
                else:
                    if aci not in invalid_cycles:
                        invalid_cycles.add(aci)
                        num_invalid += 1
            
            # Update best solution
            if num_invalid == 0:
                best_colors = colors[:]
                best_active = active_edges[:]
                best_num_edges = sum(active_edges)
                
    # If still invalid, we must remove edges to satisfy the constraint.
    # We want to maximize edges, so we remove edges that are in many invalid cycles.
    if num_invalid > 0 and time.time() < deadline - 10.0:
        # Score each edge by number of invalid cycles it's in
        edge_score = [0] * num_edges
        for ci in invalid_cycles:
            for ei in all_4cycles[ci]:
                edge_score[ei] += 1
        
        # Remove edges with highest score
        # We can do this iteratively
        while num_invalid > 0 and time.time() < deadline - 5.0:
            # Find edge with max score
            max_score = -1
            max_ei = -1
            for ei in range(num_edges):
                if active_edges[ei] and edge_score[ei] > max_score:
                    max_score = edge_score[ei]
                    max_ei = ei
            
            if max_ei == -1:
                break
                
            # Remove max_ei
            active_edges[max_ei] = False
            
            # Update invalid_cycles and scores
            new_invalid = set()
            for ci in invalid_cycles:
                if max_ei in all_4cycles[ci]:
                    # This cycle is now broken (not a 4-cycle in the graph anymore)
                    # So it's no longer "invalid" in the sense of constraint violation,
                    # because the constraint only applies to existing 4-cycles.
                    pass
                else:
                    new_invalid.add(ci)
            
            # Also check if removing this edge makes any previously valid cycle invalid?
            # No, removing an edge only removes cycles. It doesn't create new ones.
            # So we just remove all cycles containing max_ei from the invalid set.
            # And we don't need to add any new invalid cycles.
            
            invalid_cycles = new_invalid
            num_invalid = len(invalid_cycles)
            
            # Update edge scores for remaining invalid cycles
            edge_score = [0] * num_edges
            for ci in invalid_cycles:
                for ei in all_4cycles[ci]:
                    if active_edges[ei]:
                        edge_score[ei] += 1

    # Final result
    result = []
    for ei in range(num_edges):
        if active_edges[ei]:
            u, v = all_edges[ei]
            c = colors[ei]
            result.append([u, v, c])

    return result
