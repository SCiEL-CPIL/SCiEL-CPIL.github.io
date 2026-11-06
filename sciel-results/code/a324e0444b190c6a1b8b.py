"""Direct construction of a small 5-uniform hypergraph without Property B."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import sys

# Deadline guard: kill long-running work well before the evaluator limit.
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin under 120s

def construct():
    """
    Return a 5-uniform hypergraph on n vertices (5 <= n <= 26, <=200 edges)
    that has no proper 2-colouring, minimizing the number of edges.
    
    Strategy:
    1. Start with K_9^5 (126 edges).
    2. Use a fast backtracking solver with unit propagation to check 2-colorability.
    3. Greedily remove edges that do not affect non-2-colorability.
    """
    n = 9
    all_edges = [tuple(e) for e in combinations(range(n), 5)]
    edges = list(all_edges)
    
    # Fast 2-colorability checker
    def is_2_colorable(n, edges):
        if not edges:
            return True
        
        m = len(edges)
        # Precompute incidence: for each vertex, list of edge indices
        vert_edges = [[] for _ in range(n)]
        for ei, e in enumerate(edges):
            for v in e:
                vert_edges[v].append(ei)
        
        # State:
        # color: -1 (unassigned), 0, 1
        color = [-1] * n
        # counts: for each edge, count of 0s and 1s
        c0 = [0] * m
        c1 = [0] * m
        
        # Helper to assign a vertex
        def assign(v, c):
            # Returns False if conflict, True otherwise
            # Update counts for all edges containing v
            for ei in vert_edges[v]:
                if c == 0:
                    c0[ei] += 1
                else:
                    c1[ei] += 1
                
                # Check conflict: 5 of one color
                if c0[ei] == 5 or c1[ei] == 5:
                    return False
                
                # Check unit propagation: 4 of one color, 0 of other
                # If c0[ei] == 4 and c1[ei] == 0, all unassigned in edge must be 1
                # If c1[ei] == 4 and c0[ei] == 0, all unassigned in edge must be 0
                # We handle propagation in a separate loop to avoid complexity here
            color[v] = c
            return True

        def propagate():
            # Returns False if conflict, True otherwise
            # Also performs unit propagation
            changed = True
            while changed:
                changed = False
                for ei, e in enumerate(edges):
                    # Check conflict
                    if c0[ei] == 5 or c1[ei] == 5:
                        return False
                    
                    # Check unit constraints
                    unassigned_in_edge = [v for v in e if color[v] == -1]
                    
                    if c0[ei] == 4 and c1[ei] == 0:
                        # All unassigned must be 1
                        if len(unassigned_in_edge) > 0:
                            for v in unassigned_in_edge:
                                if color[v] != -1:
                                    if color[v] != 1:
                                        return False
                                else:
                                    # Assign 1
                                    color[v] = 1
                                    c1[ei] += 1
                                    # Update counts for other edges containing v
                                    for ei2 in vert_edges[v]:
                                        if ei2 != ei:
                                            c1[ei2] += 1
                                            if c1[ei2] == 5:
                                                return False
                                    changed = True
                    elif c1[ei] == 4 and c0[ei] == 0:
                        # All unassigned must be 0
                        if len(unassigned_in_edge) > 0:
                            for v in unassigned_in_edge:
                                if color[v] != -1:
                                    if color[v] != 0:
                                        return False
                                else:
                                    # Assign 0
                                    color[v] = 0
                                    c0[ei] += 1
                                    # Update counts for other edges containing v
                                    for ei2 in vert_edges[v]:
                                        if ei2 != ei:
                                            c0[ei2] += 1
                                            if c0[ei2] == 5:
                                                return False
                                    changed = True
            return True

        # Backtracking
        def backtrack():
            # Check time
            if time.time() - _START_TIME > _TIME_LIMIT:
                return True  # Assume colourable to be safe
            
            if not propagate():
                return False
            
            # Check if all assigned
            unassigned = [i for i in range(n) if color[i] == -1]
            if not unassigned:
                # Verify all edges are satisfied (mixed)
                for ei, e in enumerate(edges):
                    if c0[ei] == 5 or c1[ei] == 5:
                        return False
                return True
            
            # Choose vertex with most unassigned edges? Or just max degree?
            # Heuristic: pick vertex involved in most "active" edges (edges with unassigned vertices)
            # Simpler: max degree
            best_v = max(unassigned, key=lambda v: len(vert_edges[v]))
            
            # Try 0
            # Save state? Or use copy-on-write?
            # Since n is small, copying color, c0, c1 is cheap.
            
            # Try color 0
            color_backup = color[:]
            c0_backup = c0[:]
            c1_backup = c1[:]
            
            if assign(best_v, 0):
                if backtrack():
                    return True
            
            # Restore
            color[:] = color_backup
            c0[:] = c0_backup
            c1[:] = c1_backup
            
            # Try color 1
            if assign(best_v, 1):
                if backtrack():
                    return True
            
            return False
        
        return backtrack()

    # Greedy removal
    # We maintain a list of active edge indices
    active = list(range(len(edges)))
    
    # We iterate through the active list. If we remove an edge, we continue.
    # To be more effective, we can shuffle the order or try multiple passes.
    # Let's do a single pass first, then repeat if time permits.
    
    passes = 0
    while True:
        passes += 1
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        removed_in_pass = False
        # Iterate in reverse to allow safe removal from list? 
        # Or just build a new list.
        new_active = []
        
        for idx in active:
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                break
            
            # Try removing edge idx
            candidate_active = [i for i in active if i != idx]
            
            # If we are down to very few edges, check carefully
            if not candidate_active:
                # Can't remove if it's the last one? 
                # Actually, 0 edges is 2-colorable. 1 edge is 2-colorable.
                # We need to ensure the remaining set is NOT 2-colorable.
                # So if removing idx makes it 2-colorable, we keep idx.
                pass
            
            # Check if hypergraph with candidate_active is 2-colorable
            # We need the actual edge lists for the solver
            cand_edges = [edges[i] for i in candidate_active]
            
            # Optimization: if candidate_active is large, it's likely non-2-colorable.
            # If small, it might be 2-colorable.
            
            if not is_2_colorable(n, cand_edges):
                # It is NOT 2-colorable, so we can remove idx
                removed_in_pass = True
                # Do not add idx to new_active
            else:
                # It IS 2-colorable, so we must keep idx
                new_active.append(idx)
        
        if not removed_in_pass:
            break
        
        active = new_active
    
    final_edges = [edges[i] for i in active]
    
    # Verify
    assert 5 <= n <= 26
    assert len(final_edges) <= 200
    assert len(set(tuple(sorted(e)) for e in final_edges)) == len(final_edges)
    for e in final_edges:
        assert len(e) == 5
        assert len(set(e)) == 5
        assert all(0 <= v < n for v in e)
    
    # Final check
    if is_2_colorable(n, final_edges):
        # Fallback
        return {'n': n, 'edges': all_edges}
    
    return {'n': n, 'edges': final_edges}
# EVOLVE-BLOCK-END
