import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Construct Groetzsch Graph (Mycielski of C5)
    # Vertices 0..4 form C5
    # Vertices 5..9 are copies
    # Vertex 10 is apex
    edges_g11 = []
    # Cycle edges
    for i in range(5):
        edges_g11.append([i, (i + 1) % 5])
    # Copy edges (i, 5+i) and (i, 5+neighbor)
    for i in range(5):
        edges_g11.append([i, 5 + i])
        # Neighbors in C5 are (i-1)%5 and (i+1)%5
        n1 = (i - 1) % 5
        n2 = (i + 1) % 5
        # Edges (i, 5+n1) and (i, 5+n2) ?
        # Mycielski: (v, u') if (v,u) in E.
        # So i is connected to 5+n1 and 5+n2.
        edges_g11.append([i, 5 + n1])
        edges_g11.append([i, 5 + n2])
    # Apex edges
    for i in range(5):
        edges_g11.append([10, 5 + i])

    # Remove duplicates and sort for consistency
    edges_g11 = sorted(list(set(tuple(sorted(e)) for e in edges_g11)))
    
    # Build adjacency list for G11
    adj_g11 = [set() for _ in range(11)]
    for u, v in edges_g11:
        adj_g11[u].add(v)
        adj_g11[v].add(u)

    # Find all independent sets in G11
    # Since n=11, we can just iterate all 2^11 subsets
    independent_sets = []
    for mask in range(1 << 11):
        is_independent = True
        # Check if any edge has both endpoints in mask
        for u, v in edges_g11:
            if (mask & (1 << u)) and (mask & (1 << v)):
                is_independent = False
                break
        if is_independent:
            # Store as bitmask for fast intersection check
            independent_sets.append(mask)
            
    # We only care about non-empty sets? No, empty is allowed.
    # But we want to maximize size, so we will pick large ones.
    # Sort by size descending? Not necessary for search, but good for pruning.
    independent_sets.sort(key=lambda x: bin(x).count('1'), reverse=True)
    
    # Vertices 11..15 are the new ones.
    # Let's denote their indices as 0..4 relative to the new group.
    # We need to assign an independent set mask to each of the 5 new vertices.
    # And choose edges between them.
    
    best_edges_count = 20 # Base edges
    best_config = None

    # Iterate over all possible edge sets for the 5 new vertices
    # There are 10 possible edges.
    # We can iterate over all 2^10 = 1024 subsets.
    
    # Precompute triangle-free check for K5 subgraphs
    # Vertices 0..4. Edges are pairs.
    # Edge indices:
    # 0: (0,1), 1: (0,2), 2: (0,3), 3: (0,4)
    # 4: (1,2), 5: (1,3), 6: (1,4)
    # 7: (2,3), 8: (2,4)
    # 9: (3,4)
    
    edge_pairs = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 4)
    ]
    
    # Map edge index to pair
    # We will iterate mask_e from 0 to 1023
    
    # To speed up, we can prune.
    # Max possible added edges:
    # Max independent set size is 4.
    # 5 vertices * 4 = 20.
    # Max edges in triangle-free K5 is 6.
    # Total added = 26.
    # Total = 46.
    
    # We search for the best.
    
    # Optimization:
    # For a fixed edge mask E_new, the constraints on I_i are:
    # I_i is independent.
    # If edge (i,j) in E_new, I_i & I_j == 0.
    
    # We can iterate over assignments of independent sets?
    # Number of independent sets is likely small (hundreds?).
    # 5^N is too big if N is large.
    # But we can use backtracking with pruning.
    
    # Let's list unique independent sets by mask.
    # There are at most 2^11 = 2048, but many are not independent.
    
    # Let's try a different approach:
    # Iterate over all 1024 edge masks.
    # For each, solve the max sum of sizes problem.
    
    for e_mask in range(1024):
        if time.time() > deadline:
            break
            
        # Check if E_new is triangle-free
        # Triangles in K5: (0,1,2), (0,1,3), (0,1,4), (0,2,3), (0,2,4), (0,3,4), (1,2,3), (1,2,4), (1,3,4), (2,3,4)
        # Edge indices for these triangles:
        # (0,1,2): 0, 1, 4
        # (0,1,3): 0, 2, 5
        # (0,1,4): 0, 3, 6
        # (0,2,3): 1, 2, 7
        # (0,2,4): 1, 3, 8
        # (0,3,4): 2, 3, 9
        # (1,2,3): 4, 5, 7
        # (1,2,4): 4, 6, 8
        # (1,3,4): 5, 6, 9
        # (2,3,4): 7, 8, 9
        
        is_triangle_free = True
        # Check all 10 triangles
        triangles = [
            (0, 1, 4), (0, 2, 5), (0, 3, 6), (1, 2, 7), (1, 3, 8),
            (2, 3, 9), (4, 5, 7), (4, 6, 8), (5, 6, 9), (7, 8, 9)
        ]
        for t in triangles:
            if (e_mask & (1 << t[0])) and (e_mask & (1 << t[1])) and (e_mask & (1 << t[2])):
                is_triangle_free = False
                break
        
        if not is_triangle_free:
            continue
            
        # Now find max sum of |I_i|
        # I_i chosen from independent_sets.
        # Constraints: I_i & I_j == 0 if edge (i,j) present.
        
        # We can use backtracking on the 5 vertices.
        # Order vertices by degree in E_new to prune faster?
        # Or just fixed order 0..4.
        
        max_sum = 0
        
        # Precompute adjacency in E_new for quick check
        # adj_new[i] is bitmask of neighbors in new graph
        adj_new = [0] * 5
        for i in range(5):
            for j in range(5):
                if i == j: continue
                # Find edge index for (i,j)
                # We need a mapping
                pass
        
        # Let's just check on the fly using e_mask
        # Edge index calculation:
        # Pairs:
        # (0,1)->0, (0,2)->1, (0,3)->2, (0,4)->3
        # (1,2)->4, (1,3)->5, (1,4)->6
        # (2,3)->7, (2,4)->8
        # (3,4)->9
        
        def get_edge_idx(u, v):
            if u > v: u, v = v, u
            if u == 0:
                if v == 1: return 0
                if v == 2: return 1
                if v == 3: return 2
                if v == 4: return 3
            elif u == 1:
                if v == 2: return 4
                if v == 3: return 5
                if v == 4: return 6
            elif u == 2:
                if v == 3: return 7
                if v == 4: return 8
            elif u == 3:
                if v == 4: return 9
            return -1

        # Backtracking
        assigned = [None] * 5
        current_sum = 0
        
        def backtrack(idx, current_sum):
            nonlocal max_sum
            if idx == 5:
                if current_sum > max_sum:
                    max_sum = current_sum
                return
            
            # Try all independent sets
            # Pruning: if current_sum + 5*(11-idx) <= max_sum, return?
            # Max size is 4. So 5*4=20.
            # if current_sum + (5-idx)*4 <= max_sum: return
            
            for mask in independent_sets:
                # Check constraints with previous assigned
                valid = True
                for prev_idx in range(idx):
                    if assigned[prev_idx] is None:
                        continue
                    # Check if edge exists between idx and prev_idx
                    e_idx = get_edge_idx(idx, prev_idx)
                    if e_idx != -1 and (e_mask & (1 << e_idx)):
                        if (mask & assigned[prev_idx]) != 0:
                            valid = False
                            break
                if not valid:
                    continue
                
                assigned[idx] = mask
                backtrack(idx + 1, current_sum + bin(mask).count('1'))
                assigned[idx] = None

        backtrack(0, 0)
        
        total_edges = 20 + max_sum + bin(e_mask).count('1')
        if total_edges > best_edges_count:
            best_edges_count = total_edges
            # Store the best config
            # We need to reconstruct the edges
            # We didn't store the assignment in backtrack, only the max sum.
            # We need to store the best assignment too.
            
            # Re-run backtrack to find the assignment?
            # Or modify backtrack to store best assignment.
            
            # Let's just re-run if we find a better count?
            # Or store best_assignment in backtrack.
            
            # To save time, let's just assume we can find it again or store it.
            # I will modify the code to store the best masks.
            
    # The above loop doesn't store the actual masks for the best e_mask.
    # I will refactor to store the best masks.
    
    # Given the complexity, and that 40 is a safe baseline, 
    # and the search is small, let's just return the best found.
    
    # If the search is too slow or finds nothing better, return the 40-edge solution.
    # The 40-edge solution:
    # I_11..15 all equal to a max independent set of size 4.
    # E_new is empty.
    
    # Let's find one max independent set of size 4.
    max_ind_set = None
    for mask in independent_sets:
        if bin(mask).count('1') == 4:
            max_ind_set = mask
            break
            
    if max_ind_set is None:
        max_ind_set = independent_sets[0] if independent_sets else 0

    # Construct the 40-edge graph
    final_edges = [list(e) for e in edges_g11]
    
    # Add edges for new vertices
    # Vertices 11..15 correspond to indices 0..4 in the new group.
    # Connect each to max_ind_set vertices.
    for i in range(5):
        new_v = 11 + i
        for v in range(11):
            if max_ind_set & (1 << v):
                final_edges.append([new_v, v])
                
    # Check if we found a better solution in the search?
    # The search above only updated best_edges_count.
    # I need to properly implement the search to return the edges.
    
    # Due to the risk of bugs in the complex search within the time limit,
    # and the fact that 40 is a solid improvement over 20,
    # I will provide the 40-edge solution.
    # If I have time, I can add the search.
    # But the prompt asks for a rewrite.
    # Let's include the search properly.
    
    # Refactored Search
    best_total = 20
    best_masks = [max_ind_set] * 5
    best_e_mask = 0

    for e_mask in range(1024):
        if time.time() > deadline:
            break
        # Triangle check
        is_tf = True
        for t in triangles:
            if (e_mask & (1 << t[0])) and (e_mask & (1 << t[1])) and (e_mask & (1 << t[2])):
                is_tf = False
                break
        if not is_tf:
            continue

        # Backtracking to find max sum and store best
        assigned = [None] * 5
        max_sum = 0
        best_local_masks = [0] * 5

        def backtrack2(idx, current_sum):
            nonlocal max_sum, best_local_masks
            if idx == 5:
                if current_sum > max_sum:
                    max_sum = current_sum
                    best_local_masks = assigned[:]
                return

            for mask in independent_sets:
                valid = True
                for prev_idx in range(idx):
                    if assigned[prev_idx] is None:
                        continue
                    e_idx = get_edge_idx(idx, prev_idx)
                    if e_idx != -1 and (e_mask & (1 << e_idx)):
                        if (mask & assigned[prev_idx]) != 0:
                            valid = False
                            break
                if not valid:
                    continue
                assigned[idx] = mask
                backtrack2(idx + 1, current_sum + bin(mask).count('1'))
                assigned[idx] = None

        backtrack2(0, 0)
        
        total = 20 + max_sum + bin(e_mask).count('1')
        if total > best_total:
            best_total = total
            best_masks = best_local_masks
            best_e_mask = e_mask

    # Construct final graph from best solution
    final_edges = [list(e) for e in edges_g11]
    
    # Add edges from new vertices to G11
    for i in range(5):
        new_v = 11 + i
        mask = best_masks[i]
        for v in range(11):
            if mask & (1 << v):
                final_edges.append([new_v, v])
                
    # Add edges between new vertices
    for i in range(5):
        for j in range(5):
            if i >= j: continue
            e_idx = get_edge_idx(i, j)
            if best_e_mask & (1 << e_idx):
                final_edges.append([11 + i, 11 + j])
                
    return final_edges

