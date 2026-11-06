import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Uses bitmask representation for speed.
    Returns True if there exists a 2-coloring with no monochromatic edge.
    """
    if not edges:
        return True
        
    # Convert edges to bitmasks
    # Vertex v corresponds to bit v
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # To speed up propagation, we can group edges?
    # Or just iterate. With n<=26, edge_masks is a list of ints.
    
    # assignment is represented by mask0 and mask1
    # unassigned are ~mask0 & ~mask1 & (1<<n - 1)
    
    def propagate(mask0, mask1):
        # Returns (False, 0, 0) if contradiction, (True, mask0, mask1) otherwise
        changed = True
        while changed:
            changed = False
            for e in edge_masks:
                # Check if monochromatic 0: no vertex in e is in mask1
                if (e & mask1) == 0:
                    # All assigned vertices in e are 0 (or unassigned)
                    # If all vertices in e are assigned and in mask0, it's mono 0.
                    # If some are unassigned, it's not yet mono, but we can check unit prop.
                    
                    # Check unit prop: 4 assigned 0, 1 unassigned
                    # Assigned 0 count in e: (e & mask0).bit_count()
                    # If (e & mask1) == 0, then unassigned_in_e = e & ~mask0 & ~mask1
                    # Actually, if (e & mask1) == 0, then all assigned vertices in e are 0.
                    # Let count0 = (e & mask0).bit_count()
                    # If count0 == 5: Contradiction (mono 0)
                    # If count0 == 4: One unassigned vertex must be 1.
                    
                    count0 = (e & mask0).bit_count()
                    if count0 == 5:
                        return False, 0, 0
                    if count0 == 4:
                        # Find the unassigned vertex
                        # e & ~mask0 & ~mask1 gives the unassigned vertex in e
                        # Since count0==4 and mask1 has 0 bits in e, the remaining 1 bit is unassigned
                        unassigned_vertex = e & ~(mask0 | mask1)
                        # It should be a power of 2
                        mask1 |= unassigned_vertex
                        changed = True
                        continue # Restart loop or just continue? Changed=True handles it.
                        
                # Check if monochromatic 1: no vertex in e is in mask0
                if (e & mask0) == 0:
                    count1 = (e & mask1).bit_count()
                    if count1 == 5:
                        return False, 0, 0
                    if count1 == 4:
                        unassigned_vertex = e & ~(mask0 | mask1)
                        mask0 |= unassigned_vertex
                        changed = True
                        continue
                        
        return True, mask0, mask1

    def solve(mask0, mask1):
        # Heuristic: Pick vertex involved in most "critical" edges (count 4 assigned, 1 unassigned)
        # Or just first unassigned.
        # Let's use a slightly better heuristic:
        # Count how many edges have exactly 4 assigned of one color and 1 unassigned.
        
        unassigned_mask = ((1 << n) - 1) & ~(mask0 | mask1)
        if unassigned_mask == 0:
            return True # All assigned, and no contradiction found in propagate
        
        # Find best vertex
        # We can iterate through set bits in unassigned_mask
        # To keep it fast, let's just pick the first one or use a simple score.
        # Given n <= 26, iterating through vertices 0..n-1 is fast enough.
        
        best_v = -1
        max_score = -1
        
        # Calculate scores
        # Score[v] = number of edges where v is the unique unassigned vertex and 4 others are same color.
        # This requires checking each edge.
        # Alternatively, just pick the vertex with highest degree in the "critical" subgraph?
        # Let's stick to a simple random or first-unassigned to avoid overhead, 
        # BUT the previous improvement came from "critical edge" heuristic.
        # Let's implement the score calculation efficiently.
        
        scores = [0] * n
        
        for e in edge_masks:
            # Check if e has exactly 1 unassigned vertex
            unassigned_in_e = e & unassigned_mask
            if unassigned_in_e == 0:
                continue # Fully assigned
            if (unassigned_in_e & (unassigned_in_e - 1)) != 0:
                continue # More than 1 unassigned
            
            v = unassigned_in_e.bit_length() - 1
            
            # Check if the other 4 are same color
            # If e & mask1 == 0, then others are 0 (or unassigned, but we checked unassigned count is 1)
            # Actually, if unassigned_in_e is a single bit v, then e ^ (1<<v) are the other 4.
            # If (e & mask1) == (1<<v) is false, and (e & mask0) == (1<<v) is false...
            
            # Simpler:
            # If (e & mask1) == 0: All assigned in e are 0. Since only v is unassigned, the other 4 are 0.
            #   So v must be 1. This is a unit clause forcing 1.
            #   This was already handled by propagate? 
            #   Propagate handles count0==4. If propagate didn't fire, it means count0 != 4.
            #   Wait, if unassigned_in_e is 1 bit, and (e & mask1)==0, then count0 must be 4.
            #   So propagate SHOULD have handled it.
            #   Therefore, if we are in solve, there are no unit clauses?
            #   Not necessarily. Propagate runs to fixpoint. So if we are in solve, there are no unit clauses.
            #   So the "critical edge" heuristic for *selection* might be less critical if propagation is complete.
            #   However, picking a vertex that is in many "almost unit" edges (e.g. 3 assigned 0, 2 unassigned) might help?
            
            # Let's just pick the first unassigned vertex for speed in the solver.
            # The previous success (1.145 -> 1.167) might have been due to the specific heuristic.
            # Let's try to replicate the "critical" logic but faster?
            # Actually, if propagate is complete, the search tree is determined by branching.
            # Let's use "Fail-First": pick the vertex with the fewest remaining values? No, binary.
            # Pick the vertex involved in the most edges?
            
            pass # We will just use the first unassigned vertex to keep solver fast.
            # If performance drops, we can add heuristic back.
            
        # Optimization: Find first unassigned
        # We can extract the lowest set bit
        v = (unassigned_mask & -unassigned_mask).bit_length() - 1
        
        # Try 0
        new_mask0 = mask0 | (1 << v)
        ok, m0, m1 = propagate(new_mask0, mask1)
        if ok and solve(m0, m1):
            return True
            
        # Try 1
        new_mask1 = mask1 | (1 << v)
        ok, m0, m1 = propagate(mask0, new_mask1)
        if ok and solve(m0, m1):
            return True
            
        return False

    ok, m0, m1 = propagate(0, 0)
    if not ok:
        return False
        
    return solve(m0, m1)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Search n from 10 to 26
    # Smaller n -> fewer combinations, easier to find small non-colorable?
    # But we need at least enough vertices to support the edges.
    
    for n in range(10, 27):
        if time.time() > deadline:
            break
            
        restarts = 50 # Increase restarts since solver is faster
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            
            # Start with a moderate number of edges
            k = min(len(all_possible), 40 + random.randint(0, 20))
            edges = [list(e) for e in all_possible[:k]]
            
            edge_set = set(tuple(e) for e in edges)
            
            # Ensure non-2-colorable
            # If colorable, add edges
            while is_2_colorable_fast(n, edges):
                if time.time() > deadline:
                    break
                if len(edges) < len(all_possible):
                    # Pick random edge
                    while True:
                        new_edge = tuple(random.choice(all_possible))
                        if new_edge not in edge_set:
                            edges.append(list(new_edge))
                            edge_set.add(new_edge)
                            break
                else:
                    break
            
            if time.time() > deadline:
                break
                
            # Verify non-2-colorable
            if is_2_colorable_fast(n, edges):
                continue
                
            # Greedy Deletion
            current_edges = set(tuple(e) for e in edges)
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                removal_list = list(current_edges)
                random.shuffle(removal_list)
                
                for edge_tuple in removal_list:
                    if time.time() > deadline:
                        break
                        
                    if edge_tuple not in current_edges:
                        continue
                        
                    test_edges = current_edges - {edge_tuple}
                    test_list = [list(e) for e in test_edges]
                    
                    if not is_2_colorable_fast(n, test_list):
                        current_edges = test_edges
                        improved = True
                        break # Restart pass to re-shuffle and check new state
                        
            final_edges = [list(e) for e in current_edges]
            if not is_2_colorable_fast(n, final_edges):
                if len(final_edges) < best_count:
                    best_count = len(final_edges)
                    best_edges = final_edges
                    best_n = n
                    print(f"Found better: n={best_n}, edges={best_count}")
                    
    # Fallback: Complete graph on 9 (K_9^5) is not 2-colorable?
    # K_9^5 has C(9,5)=126 edges.
    # Is it 2-colorable?
    # If we color 5 vertices 0 and 4 vertices 1, any edge with 5 0s is mono.
    # In K_9^5, there is an edge consisting of 5 specific 0s. So yes, not 2-colorable.
    # But we want MIN edges.
    
    if best_edges is None:
        # Use a known small construction or K_9^5
        # K_9^5 is 126 edges.
        # Let's try to find a smaller one quickly if time permits, else return K_9^5?
        # No, the loop above should find something better than 126.
        # If not, return K_9^5.
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        
    return {'n': best_n, 'edges': best_edges}
