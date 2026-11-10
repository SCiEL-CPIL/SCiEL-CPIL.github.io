import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115  # Leave 5s margin
    
    best_edges = None
    best_n = None
    min_edges_found = float('inf')
    
    def is_non_2_colorable(n, edges):
        # Fast DPLL solver using bitmasks for state
        m = len(edges)
        if m == 0:
            return False
        
        # Precompute bitmasks for each edge
        edge_masks = [0] * m
        for i, e in enumerate(edges):
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks[i] = mask
            
        # Adjacency: vertex -> list of edge indices
        v_to_e = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_to_e[v].append(i)

        def dfs(assign_mask, forbid0_mask, forbid1_mask):
            # assign_mask: bits set for assigned vertices
            # forbid0_mask: bits set for vertices that cannot be 0
            # forbid1_mask: bits set for vertices that cannot be 1
            
            # Check for conflict: vertex assigned to 0 but in forbid0? 
            # We handle this during assignment, but let's check consistency.
            # Actually, if we maintain invariants, conflicts happen when we try to assign.
            
            # Propagation:
            # If a vertex is assigned to 0, all vertices in edges with it that are unassigned 
            # and would make the edge mono-0 if assigned 0... 
            # Wait, simpler: 
            # If an edge has 4 vertices assigned to 0, the 5th MUST be 1.
            # If an edge has 4 vertices assigned to 1, the 5th MUST be 0.
            # If an edge has 5 vertices assigned to 0 (or 1), UNSAT.
            
            # We can check propagation by iterating edges.
            # But iterating all edges at each step is O(m).
            # n <= 26, m <= 200.
            
            # Let's find unassigned vertices.
            unassigned = ~assign_mask & ((1 << n) - 1)
            
            if unassigned == 0:
                # All assigned. Check if any edge is monochromatic?
                # If we reached here, propagation didn't kill it.
                # But we must ensure no edge is mono.
                # Propagation ensures that if 4 are same, 5th is forced to other.
                # So if all assigned, no edge can have 5 same?
                # Wait. If an edge has 5 same, c0 or c1 would be 5.
                # Propagation returns False if c0==5 or c1==5.
                # So if we reach here, it is a valid coloring.
                return True
            
            # Heuristic: Pick an unassigned vertex.
            # Prefer vertices involved in "tight" edges (edges with many assigned).
            # Let's just pick the first unassigned for simplicity, or one with high degree.
            # For speed, let's pick the first unassigned bit.
            v = (unassigned & -unassigned).bit_length() - 1
            
            # Try assigning v to 0
            if not (forbid0_mask & (1 << v)):
                new_assign = assign_mask | (1 << v)
                new_forbid0 = forbid0_mask
                new_forbid1 = forbid1_mask
                ok = True
                
                # Check edges containing v
                for e_idx in v_to_e[v]:
                    mask = edge_masks[e_idx]
                    # Count how many vertices in this edge are assigned to 0
                    # We don't have a separate "assigned to 0" mask.
                    # We have assign_mask.
                    # We need to know which assigned vertices are 0 vs 1.
                    # This bitmask approach requires tracking colors separately or 
                    # deriving them.
                    # If we only track assign_mask, we don't know colors of assigned vertices.
                    # So we need to track colors.
                    
                    # Let's restart with a simpler state representation if bitmasks for colors are complex.
                    # State: (colors_mask_0, colors_mask_1) ? 
                    # colors_mask_0: bits set for vertices colored 0.
                    # colors_mask_1: bits set for vertices colored 1.
                    # assign_mask = colors_mask_0 | colors_mask_1.
                    
                    # This is better.
                    pass
                # The above logic is flawed because we don't track colors in the state passed to dfs.
                # Let's use state: (mask0, mask1) where mask0 has bits for 0, mask1 for 1.
                # Unassigned = ~(mask0 | mask1).
                
                # Let's rewrite dfs to use mask0, mask1.
                pass

            # This is getting complicated to inline. Let's define a cleaner recursive function.
            return False # Placeholder

        # Let's implement the clean version.
        
        def dfs_real(mask0, mask1):
            # mask0: bits for vertices colored 0
            # mask1: bits for vertices colored 1
            # assign_mask = mask0 | mask1
            # unassigned = ~(mask0 | mask1) & ((1 << n) - 1)
            
            # Check for conflicts:
            # If any edge has all 5 bits in mask0 -> UNSAT
            # If any edge has all 5 bits in mask1 -> UNSAT
            # Also, if any edge has 4 bits in mask0 and the 5th is unassigned, the 5th MUST be 1.
            # If any edge has 4 bits in mask1 and the 5th is unassigned, the 5th MUST be 0.
            
            # We perform propagation in a loop.
            # Since n is small, we can just check all edges.
            
            changed = True
            while changed:
                changed = False
                for e_idx in range(m):
                    mask = edge_masks[e_idx]
                    c0 = bin(mask & mask0).count('1')
                    c1 = bin(mask & mask1).count('1')
                    
                    if c0 == 5 or c1 == 5:
                        return False # Conflict
                    
                    if c0 == 4:
                        # Find the unassigned vertex in this edge
                        unassigned_in_edge = mask & ~(mask0 | mask1)
                        if unassigned_in_edge:
                            # There is exactly one unassigned vertex (since c0=4, c1=0 or c1=1? 
                            # If c0=4, c1 can be 0 or 1. If c1=1, then 4+1=5, all assigned. 
                            # So if c0=4 and there is an unassigned vertex, c1 must be 0? 
                            # Wait. If c0=4 and c1=1, all 5 are assigned. 
                            # If c0=4 and c1=0, 1 is unassigned.
                            # If c0=4 and c1=0, the unassigned one must be 1.
                            # If c0=4 and c1=1, no unassigned.
                            # So we only force if there is an unassigned vertex.
                            
                            # Is it possible to have c0=4, c1=0, and 1 unassigned? Yes.
                            # Is it possible to have c0=4, c1=1? Yes, then all assigned.
                            
                            # So if unassigned_in_edge is non-zero, we force it to 1.
                            # There should be only 1 such vertex.
                            # Let's verify: mask has 5 bits. mask0 has 4 of them. mask1 has 0 or 1.
                            # If mask1 has 1, then 4+1=5, all assigned. unassigned_in_edge is 0.
                            # If mask1 has 0, then 4+0=4, 1 unassigned. unassigned_in_edge has 1 bit.
                            
                            v_bit = unassigned_in_edge
                            if mask1 & v_bit:
                                return False # Already forced to 1? No, if it's in mask1, it's assigned.
                                             # But we said unassigned_in_edge is ~(mask0|mask1). So it's not in mask1.
                            mask1 |= v_bit
                            changed = True
                            
                    elif c1 == 4:
                        unassigned_in_edge = mask & ~(mask0 | mask1)
                        if unassigned_in_edge:
                            v_bit = unassigned_in_edge
                            if mask0 & v_bit:
                                return False
                            mask0 |= v_bit
                            changed = True
                            
            # After propagation, check for conflicts again?
            # The loop above returns False if conflict.
            # But what if we assign a vertex and it creates a conflict immediately?
            # The loop checks c0==5 or c1==5.
            
            unassigned = ~(mask0 | mask1) & ((1 << n) - 1)
            if unassigned == 0:
                return True # All assigned, no conflict -> Valid coloring found
            
            # Choose a vertex to branch on.
            # Heuristic: pick vertex in an edge with max assigned count?
            # Simple: first unassigned.
            v = (unassigned & -unassigned).bit_length() - 1
            v_bit = 1 << v
            
            # Try 0
            if not (mask1 & v_bit): # If not forced to 1 (which would mean it's in mask1, but it's unassigned so it's not)
                # Actually, if it's unassigned, it's not in mask0 or mask1.
                # So we can always try 0 and 1 unless propagation forced it.
                # Wait, if propagation forced it, it would be in mask0 or mask1.
                # Since it's unassigned, it's not.
                
                if dfs_real(mask0 | v_bit, mask1):
                    return True
            
            # Try 1
            if dfs_real(mask0, mask1 | v_bit):
                return True
            
            return False

        if dfs_real(0, 0):
            return False # 2-colorable
        else:
            return True # Non-2-colorable

    # Minimization
    def minimize(n, edges):
        current_edges = [list(e) for e in edges]
        improved = True
        while improved:
            improved = False
            for i in range(len(current_edges)):
                if time.time() > deadline:
                    return current_edges
                # Try removing edge i
                test_edges = [e for j, e in enumerate(current_edges) if j != i]
                if is_non_2_colorable(n, test_edges):
                    current_edges = test_edges
                    improved = True
                    break # Restart loop with new list
        return current_edges

    # Search
    # Try n from 8 to 12
    for n in range(8, 13):
        if time.time() > deadline:
            break
            
        # Random search for a small non-2-colorable hypergraph
        attempts = 20
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            # Pick a random m between 30 and 60
            m = random.randint(30, 60)
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            edges = [list(e) for e in all_edges[:m]]
            
            if is_non_2_colorable(n, edges):
                min_edges = minimize(n, edges)
                if len(min_edges) < min_edges_found:
                    min_edges_found = len(min_edges)
                    best_edges = min_edges
                    best_n = n
                    
    if best_edges is None:
        # Fallback: Return a known large one? Or just K_9?
        # K_9 has 126 edges.
        # Is there a smaller fallback?
        # The task says "best known is 51".
        # If we fail, we return K_9?
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
