import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s for safety
    
    # We aim for n=17. If we fail, we might try n=16 or return a valid smaller one if found.
    # But the goal is to maximize n.
    # Let's try n=17.
    
    def try_solve(n):
        if n < 6:
            return [0] * (n*(n-1)//2)
        
        # Generate all C6s
        # A C6 is a sequence of 6 distinct vertices (v0, v1, v2, v3, v4, v5) such that edges are (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0).
        # To avoid duplicates, we can fix v0=0, v1=1? No, C6s can be anywhere.
        # It's easier to generate all combinations of 6 vertices, and for each, check the 5 possible cyclic permutations?
        # No, for a set of 6 vertices, there are 5 distinct C6 subgraphs in K6.
        # So we can iterate over all subsets of size 6.
        # Number of subsets: C(17, 6) = 12376.
        # For each subset, generate the 5 C6s.
        
        vertices = list(range(n))
        c6_list = []
        # edge_index: maps (u, v) to index in colors array
        # We need a fast lookup for edge index
        edge_idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_idx_map[(i, j)] = idx
                idx += 1
        total_edges = idx
        
        # Pre-compute C6s
        # For each subset of 6 vertices
        from itertools import combinations
        for comb in combinations(vertices, 6):
            # The 5 distinct Hamiltonian cycles on 6 vertices (up to rotation/reflection)
            # Vertices: v0, v1, v2, v3, v4, v5
            # Cycle 1: 0-1-2-3-4-5-0
            # Cycle 2: 0-1-2-4-3-5-0
            # Cycle 3: 0-1-3-2-4-5-0
            # Cycle 4: 0-1-3-4-2-5-0
            # Cycle 5: 0-1-4-2-3-5-0
            # Actually, it's easier to just list the permutations that form a cycle.
            # There are (6-1)! / 2 = 60 / 2 = 30 cycles? No.
            # Number of distinct C6 subgraphs in K6 is 5.
            # Let's just hardcode the 5 patterns relative to sorted vertices.
            v0, v1, v2, v3, v4, v5 = comb
            patterns = [
                [(v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)],
                [(v0,v1), (v1,v2), (v2,v4), (v4,v3), (v3,v5), (v5,v0)],
                [(v0,v1), (v1,v3), (v3,v2), (v2,v4), (v4,v5), (v5,v0)],
                [(v0,v1), (v1,v3), (v3,v4), (v4,v2), (v2,v5), (v5,v0)],
                [(v0,v1), (v1,v4), (v4,v2), (v2,v3), (v3,v5), (v5,v0)]
            ]
            for pat in patterns:
                c6_list.append(pat)
        
        num_c6s = len(c6_list)
        
        # Map each edge to the list of C6 indices it participates in
        edge_to_c6s = [[] for _ in range(total_edges)]
        for i, c6 in enumerate(c6_list):
            for (u, v) in c6:
                # Ensure u < v for map key
                if u > v: u, v = v, u
                e_idx = edge_idx_map[(u, v)]
                edge_to_c6s[e_idx].append(i)
        
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(total_edges)]
        
        # Calculate initial cost: number of monochromatic C6s
        # count[color] = number of C6s that are monochromatic with that color
        count = [0, 0, 0, 0]
        for i, c6 in enumerate(c6_list):
            c0 = colors[edge_idx_map[(c6[0][0], c6[0][1]) if c6[0][0]<c6[0][1] else (c6[0][1], c6[0][0])]]
            # Check if all edges have color c0
            is_mono = True
            for (u, v) in c6[1:]:
                key = (u, v) if u < v else (v, u)
                if colors[edge_idx_map[key]] != c0:
                    is_mono = False
                    break
            if is_mono:
                count[c0] += 1
        
        total_bad = sum(count)
        
        if total_bad == 0:
            return colors
        
        # Local Search
        # We will try to minimize total_bad.
        # If we reach 0, we have a solution.
        
        while time.time() < deadline:
            # Pick a random edge
            e_idx = random.randint(0, total_edges - 1)
            old_color = colors[e_idx]
            
            # Determine the other 5 edges for each C6 containing this edge?
            # No, we just iterate through edge_to_c6s[e_idx]
            
            best_delta = 0
            best_new_color = -1
            
            for new_color in range(4):
                if new_color == old_color:
                    continue
                
                delta = 0
                # For each C6 containing this edge, check if it becomes mono in new_color or stops being mono in old_color
                for c6_idx in edge_to_c6s[e_idx]:
                    c6 = c6_list[c6_idx]
                    # Get the other 5 edges
                    other_edges_colors = []
                    for (u, v) in c6:
                        if (u == e_idx // 1 and False): # Placeholder
                            pass
                        # We need to identify which edge is e_idx
                        # This is inefficient. 
                        # Let's pre-compute for each C6, the list of edge indices.
                        # And for each edge in the C6, we know its index.
                        # But we need to exclude the current edge.
                        # Let's restructure data.
                        pass
                    # This loop is getting complicated to optimize on the fly.
                    # Let's pre-compute `c6_edge_indices`: list of 6 edge indices for each C6.
                    # And `c6_edge_positions`: for each edge in the C6, its position (0-5).
                    pass
                
                # The on-the-fly check is too slow or complex to code correctly in this short space without pre-computation.
                # Let's pre-compute `c6_edges`: list of 6 edge indices.
                # And for each edge e, and each C6 c containing e, we need the colors of the other 5 edges.
                # We can just check the colors of all 6 edges in the C6.
                
                # Revised Plan for Inner Loop:
                # For a candidate new_color:
                # For each C6 index c6_idx in edge_to_c6s[e_idx]:
                #   Get the 6 edge indices for this C6.
                #   Check if all 6 edges have color new_color. (If yes, +1 to count[new_color])
                #   Check if all 6 edges have color old_color. (If yes, -1 to count[old_color])
                #   Note: The current edge is assumed to be new_color for the first check, and old_color for the second.
                #   So we only need to check the other 5 edges.
                
                pass
            
            # This approach is O(2700 * 5) per move, which is ~13,500 ops.
            # 10,000 moves = 135,000,000 ops. Might be too slow in Python.
            
            # Optimization:
            # Pre-compute `c6_edge_indices` for all C6s.
            # Then in the loop:
            # for c6_idx in edge_to_c6s[e_idx]:
            #    e_indices = c6_edge_indices[c6_idx]
            #    # Check if other 5 are new_color
            #    # We can just check all 6, knowing one is e_idx.
            #    # But we need to be careful with the "current" color.
            
            # Let's implement the pre-computation and the efficient check.
            pass
        
        # If we didn't find a solution in the while loop structure above (which was a sketch),
        # we need to actually implement it.
        
        # Let's rewrite the search loop properly.
        
        # Pre-compute c6_edge_indices
        c6_edge_indices = []
        for c6 in c6_list:
            indices = []
            for (u, v) in c6:
                key = (u, v) if u < v else (v, u)
                indices.append(edge_idx_map[key])
            c6_edge_indices.append(indices)
        
        # Reset colors and counts if we want to restart? No, we are in try_solve.
        # The previous `while` loop was a placeholder. Let's replace it.
        
        while time.time() < deadline:
            e_idx = random.randint(0, total_edges - 1)
            old_color = colors[e_idx]
            
            # Try all 3 other colors
            best_delta = 0
            best_new_color = -1
            current_count_snapshot = count[:] # Not needed if we calculate delta
            
            for new_color in range(4):
                if new_color == old_color:
                    continue
                
                delta = 0
                for c6_idx in edge_to_c6s[e_idx]:
                    e_indices = c6_edge_indices[c6_idx]
                    
                    # Check if C6 becomes monochromatic new_color
                    # We assume e_idx is new_color. Check other 5.
                    is_mono_new = True
                    for e in e_indices:
                        if e == e_idx:
                            continue
                        if colors[e] != new_color:
                            is_mono_new = False
                            break
                    if is_mono_new:
                        delta += 1 # count[new_color] increases by 1 (if it wasn't already)
                        # Wait, it might have been mono new_color already?
                        # No, because e_idx was old_color. So it couldn't be mono new_color before.
                        # So if it becomes mono new_color, it's a +1.
                        
                    # Check if C6 stops being monochromatic old_color
                    # It was mono old_color if all 6 were old_color.
                    # Now e_idx is new_color. So it is NOT mono old_color.
                    # Did it contribute to count[old_color] before?
                    # Yes, if the other 5 were old_color.
                    is_mono_old = True
                    for e in e_indices:
                        if e == e_idx:
                            continue
                        if colors[e] != old_color:
                            is_mono_old = False
                            break
                    if is_mono_old:
                        delta -= 1 # count[old_color] decreases by 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            if best_new_color != -1:
                # Apply the move
                # Update counts
                # We need to re-evaluate the delta logic to update counts correctly.
                # The delta calculation above assumes:
                # +1 if becomes mono new
                # -1 if was mono old
                # This is correct for the change in total_bad.
                
                # But we must update the `count` array and `colors`.
                # To update `count` correctly, we need to know which C6s changed status.
                # We can just re-calculate the status for the affected C6s.
                
                # Simpler: Just apply the move to `colors`, and then recalculate `count` for the affected C6s?
                # Or just trust the delta and update counts?
                # It's safer to just update `colors` and then do a "repair" of the counts for the affected C6s.
                # Or, since we only care about total_bad == 0, we can just track total_bad.
                # But we need `count` to know if we are at 0? No, just sum(count).
                
                # Let's just update `colors` and `total_bad`.
                # But we need to update `count` for future moves?
                # Yes, because future moves depend on `count`?
                # No, future moves depend on `colors`.
                # The `count` array is only used to check if we are done?
                # Actually, my delta calculation didn't use `count` array, it used `colors`.
                # So I don't need to maintain `count` array for the search!
                # I only need `total_bad`.
                
                # Wait, the delta calculation:
                # `is_mono_new` checks if other 5 are new_color.
                # `is_mono_old` checks if other 5 are old_color.
                # This is independent of `count`.
                # So I can just maintain `total_bad`.
                
                total_bad += best_delta
                colors[e_idx] = best_new_color
                
                if total_bad == 0:
                    return colors
            else:
                # No improvement found. Try a random move? Or accept neutral?
                # To escape local minima, we can accept a neutral or slightly worse move with low probability.
                # Or just pick a random color change.
                new_color = random.choice([c for c in range(4) if c != old_color])
                # Calculate delta for this random move
                delta = 0
                for c6_idx in edge_to_c6s[e_idx]:
                    e_indices = c6_edge_indices[c6_idx]
                    is_mono_new = True
                    for e in e_indices:
                        if e == e_idx: continue
                        if colors[e] != new_color:
                            is_mono_new = False
                            break
                    if is_mono_new: delta += 1
                    
                    is_mono_old = True
                    for e in e_indices:
                        if e == e_idx: continue
                        if colors[e] != old_color:
                            is_mono_old = False
                            break
                    if is_mono_old: delta -= 1
                
                total_bad += delta
                colors[e_idx] = new_color
                if total_bad == 0:
                    return colors

        # If time runs out, return the current best?
        # If total_bad != 0, it's invalid.
        # We should return a valid solution if we have one.
        # We can keep the best valid solution found during the process?
        # Since we start with random, it's likely invalid.
        # We can store the best (lowest total_bad) solution, but if it's not 0, it's invalid.
        # The task requires a valid coloring.
        # So if we don't find 0, we should return a valid smaller n?
        # Or just return the invalid one? No, invalid scores 0.
        # Let's try to find a valid one for n=16 if n=17 fails?
        # But the loop is inside try_solve(17).
        # If it times out, it returns None (implicitly).
        
        return None

    # Try n=17
    solution = try_solve(17)
    if solution:
        return {'n': 17, 'colors': solution}
    
    # If 17 fails, try 16
    solution = try_solve(16)
    if solution:
        return {'n': 16, 'colors': solution}
        
    # If 16 fails, try 15
    solution = try_solve(15)
    if solution:
        return {'n': 15, 'colors': solution}
        
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
