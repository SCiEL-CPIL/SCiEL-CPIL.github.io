import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    num_edges = len(edges)
    
    # Convert edges to bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # For small n, use a vectorized-ish approach or simple loop.
    # The constraint is "no monochromatic edge".
    # An edge is monochromatic if (mask & color_mask) == 0 OR (mask & color_mask) == mask.
    
    # Heuristic: If n is small enough, we can sometimes speed up, but 2^n is exponential.
    # Let's stick to a robust backtracker for all sizes but optimize the state.
    
    # Precompute vertex -> list of edges
    vertex_edges = [[] for _ in range(n)]
    for i, em in enumerate(edge_masks):
        for v in range(n):
            if em & (1 << v):
                vertex_edges[v].append(i)
    
    # State
    c0 = [0] * num_edges
    c1 = [0] * num_edges
    color = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for i in range(num_edges):
                # Check for immediate conflict
                if c0[i] == 5 or c1[i] == 5:
                    return False
                
                # Unit propagation
                # If 4 are 0, the 5th MUST be 1.
                if c0[i] == 4 and c1[i] == 0:
                    # Find the unassigned vertex in this edge
                    unassigned_v = -1
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            unassigned_v = v
                            break
                    if unassigned_v != -1:
                        color[unassigned_v] = 1
                        c1[i] += 1
                        # Update other edges
                        for other_e in vertex_edges[unassigned_v]:
                            c1[other_e] += 1
                        changed = True
                        # Check if this caused a conflict in another edge
                        if c1[i] == 5: # Should not happen if we checked before, but safe
                            return False
                        # Immediate check on the edge we just updated
                        # Actually, the loop continues, so we will check all edges.
                elif c1[i] == 4 and c0[i] == 0:
                    unassigned_v = -1
                    for v in range(n):
                        if color[v] == -1 and (edge_masks[i] & (1 << v)):
                            unassigned_v = v
                            break
                    if unassigned_v != -1:
                        color[unassigned_v] = 0
                        c0[i] += 1
                        for other_e in vertex_edges[unassigned_v]:
                            c0[other_e] += 1
                        changed = True
        return True

    def backtrack():
        # Find the most constrained edge
        best_edge = -1
        best_unassigned = 6
        best_max_color = 0
        
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
            unassigned = 5 - c0[i] - c1[i]
            max_color = max(c0[i], c1[i])
            
            # Prioritize edges with fewer unassigned vertices
            if unassigned < best_unassigned or (unassigned == best_unassigned and max_color > best_max_color):
                best_unassigned = unassigned
                best_max_color = max_color
                best_edge = i
        
        if best_edge == -1:
            # All edges fully assigned, check for final conflict
            for i in range(num_edges):
                if c0[i] == 5 or c1[i] == 5:
                    return False
            return True
            
        if best_unassigned == 0:
            return False # Should be caught above, but safe

        # Get unassigned vertices in best_edge
        candidates = []
        for v in range(n):
            if color[v] == -1 and (edge_masks[best_edge] & (1 << v)):
                candidates.append(v)
                
        if not candidates:
            return True # Edge satisfied with 0 unassigned? No, if best_unassigned > 0, there must be candidates.
                        # If best_unassigned == 0, we returned False.
        
        # Try assigning each candidate
        for v in candidates:
            # Try color 0
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            
            color[v] = 0
            ok = True
            for e_idx in vertex_edges[v]:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    ok = False
                    break
            
            if ok and propagate():
                if backtrack():
                    return True
            
            # Unwind
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
            
            # Try color 1
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            
            color[v] = 1
            ok = True
            for e_idx in vertex_edges[v]:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    ok = False
                    break
            
            if ok and propagate():
                if backtrack():
                    return True
            
            # Unwind
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
            
        return False

    return backtrack()

def construct():
    start_time = time.time()
    # Hard limit 120s, leave 10s buffer for overhead/return
    deadline = start_time + 110
    
    best_edges = None
    best_n = 0
    best_m = 10**9
    
    # Strategy:
    # 1. Start with complete 5-uniform hypergraphs for n=9..14
    # 2. Greedily remove edges using randomized order to minimize count while keeping non-2-colorable
    # 3. Repeat with different random seeds to escape local optima
    
    for n in range(9, 15):
        if time.time() > deadline:
            break
            
        all_possible = list(combinations(range(n), 5))
        
        # Check if complete hypergraph is non-2-colorable
        # If it is 2-colorable, skip this n (though complete usually isn't for n>=9? 
        # K_9^5 is not 2-colorable. K_8^5 is 2-colorable? 
        # Actually, we just check it.)
        if is_2_colorable_fast(n, [list(e) for e in all_possible]):
            continue
            
        current_edges = [list(e) for e in all_possible]
        current_m = len(current_edges)
        
        # Multiple restarts for each n
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Shuffle order for removal
            order = list(range(len(current_edges)))
            random.shuffle(order)
            
            improved = True
            while improved:
                improved = False
                for idx in order:
                    if time.time() > deadline:
                        break
                    if idx >= len(current_edges):
                        continue
                        
                    # Try removing edge at index idx
                    # Optimization: Instead of creating new list every time, we can swap, but list deletion is O(n).
                    # Given m <= 200, O(m^2) is fine.
                    new_edges = current_edges[:idx] + current_edges[idx+1:]
                    
                    if not is_2_colorable_fast(n, new_edges):
                        current_edges = new_edges
                        current_m = len(current_edges)
                        improved = True
                        # Reshuffle remaining indices to try different order
                        order = list(range(len(current_edges)))
                        random.shuffle(order)
                        break # Restart inner loop with new state
        
        if current_m < best_m:
            best_m = current_m
            best_edges = current_edges[:]
            best_n = n
    
    # If we found a very good solution, try to push further with local search on the best one
    if best_edges is not None and best_m > 30:
        n = best_n
        current_edges = best_edges[:]
        current_m = len(current_edges)
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            order = list(range(len(current_edges)))
            random.shuffle(order)
            
            improved = True
            while improved:
                improved = False
                for idx in order:
                    if time.time() > deadline:
                        break
                    if idx >= len(current_edges):
                        continue
                    new_edges = current_edges[:idx] + current_edges[idx+1:]
                    if not is_2_colorable_fast(n, new_edges):
                        current_edges = new_edges
                        current_m = len(current_edges)
                        improved = True
                        order = list(range(len(current_edges)))
                        random.shuffle(order)
                        break
        
        if current_m < best_m:
            best_m = current_m
            best_edges = current_edges[:]
            best_n = n
            
    # Fallback if nothing found (should not happen for n=9)
    if best_edges is None:
        # Construct a known non-2-colorable hypergraph, e.g., Fano plane dual or just K_9^5 subset
        # K_9^5 is 126 edges. We can just return that if search fails.
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
