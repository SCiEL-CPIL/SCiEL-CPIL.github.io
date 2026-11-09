import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    # Map vertex to color: -1 (unassigned), 0, 1
    colors = [-1] * n
    
    # Precompute for each vertex, which edges it belongs to
    vertex_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            vertex_edges[v].append(i)
            
    # Count of 0s and 1s in each edge
    count_0 = [0] * len(edges)
    count_1 = [0] * len(edges)
    
    def propagate(idx, color):
        # Assign colors[idx] = color
        # Returns False if contradiction, True if okay
        if colors[idx] != -1:
            return colors[idx] == color
        
        colors[idx] = color
        for e_idx in vertex_edges[idx]:
            if color == 0:
                count_0[e_idx] += 1
                if count_0[e_idx] == 5:
                    return False # Monochromatic 0
            else:
                count_1[e_idx] += 1
                if count_1[e_idx] == 5:
                    return False # Monochromatic 1
            
            # Unit propagation check
            # If 4 are 0, the last must be 1
            # If 4 are 1, the last must be 0
            if count_0[e_idx] == 4:
                # Find the unassigned vertex in this edge
                unassigned = None
                for v in edges[e_idx]:
                    if colors[v] == -1:
                        if unassigned is not None:
                            return False # Should not happen if consistent
                        unassigned = v
                if unassigned is not None:
                    if not propagate(unassigned, 1):
                        return False
            elif count_1[e_idx] == 4:
                unassigned = None
                for v in edges[e_idx]:
                    if colors[v] == -1:
                        if unassigned is not None:
                            return False
                        unassigned = v
                if unassigned is not None:
                    if not propagate(unassigned, 0):
                        return False
        return True

    def backtrack():
        # Find unassigned vertex
        # Heuristic: pick vertex involved in most constraints? Or just first?
        # For small n, first is fine.
        idx = -1
        for i in range(n):
            if colors[i] == -1:
                idx = i
                break
        
        if idx == -1:
            return True # All assigned, no contradiction found
        
        # Try 0
        if propagate(idx, 0):
            if backtrack():
                return True
            # Undo
            for e_idx in vertex_edges[idx]:
                count_0[e_idx] -= 1
            colors[idx] = -1
            # Note: Undoing propagation is tricky. 
            # Standard backtracking usually copies state or uses a trail.
            # Given the small size, copying state is safer.
        
        # Try 1
        if propagate(idx, 1):
            if backtrack():
                return True
            # Undo
            for e_idx in vertex_edges[idx]:
                count_1[e_idx] -= 1
            colors[idx] = -1
            
        return False

    # The undo logic above is flawed because propagate might have triggered other assignments.
    # Let's use a simpler, state-copying backtracker for correctness and simplicity, 
    # which is fast enough for n<=15, E<=50.
    
    def solve(state_colors, state_count_0, state_count_1):
        # Find unassigned
        idx = -1
        for i in range(n):
            if state_colors[i] == -1:
                idx = i
                break
        if idx == -1:
            return True
            
        # Try 0
        new_c = state_colors.copy()
        new_c0 = state_count_0.copy()
        new_c1 = state_count_1.copy()
        if assign(idx, 0, new_c, new_c0, new_c1):
            if solve(new_c, new_c0, new_c1):
                return True
        
        # Try 1
        new_c = state_colors.copy()
        new_c0 = state_count_0.copy()
        new_c1 = state_count_1.copy()
        if assign(idx, 1, new_c, new_c0, new_c1):
            if solve(new_c, new_c0, new_c1):
                return True
                
        return False

    def assign(idx, color, c, c0, c1):
        if c[idx] != -1:
            return c[idx] == color
        c[idx] = color
        for e_idx in vertex_edges[idx]:
            if color == 0:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    return False
            else:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    return False
            
            # Propagate
            if c0[e_idx] == 4:
                for v in edges[e_idx]:
                    if c[v] == -1:
                        if not assign(v, 1, c, c0, c1):
                            return False
            elif c1[e_idx] == 4:
                for v in edges[e_idx]:
                    if c[v] == -1:
                        if not assign(v, 0, c, c0, c1):
                            return False
        return True

    return solve([-1]*n, [0]*len(edges), [0]*len(edges))

def construct():
    # We want to find a small non-2-colorable 5-uniform hypergraph.
    # Known bounds: m(5) <= 51.
    # We try to find one with fewer edges via random search.
    
    best_edges = None
    best_count = 1000
    
    # Try n from 9 to 15
    for n in range(9, 16):
        all_edges = list(combinations(range(n), 5))
        # We want to find a subset of size E that is not 2-colorable.
        # Start E from 29 (known lower bound-ish) up to 51.
        for E in range(29, 52):
            # Limit random tries to avoid timeout
            # More tries for smaller E? No, smaller E is harder to find non-colorable.
            # Actually, as E increases, it becomes EASIER to be non-colorable.
            # So we should start with small E and increase.
            
            # If E is small, probability of being non-colorable is low.
            # If E is large, probability is high.
            
            # Heuristic: try a few random subsets
            max_tries = 500
            for _ in range(max_tries):
                random.shuffle(all_edges)
                candidate = [list(e) for e in all_edges[:E]]
                
                if not is_2_colorable(n, candidate):
                    if E < best_count:
                        best_count = E
                        best_edges = candidate
                        # We can break early if we find a very good one, 
                        # but let's try to find even smaller ones?
                        # The loop goes E=29, 30... so if we find one at 30, 
                        # we continue to next n or break?
                        # We want the GLOBAL minimum.
                        # So we record it and continue searching for smaller E in other n's?
                        # But E loop is ascending. So if we find one at 30, 
                        # we know 29 didn't work (for this n). 
                        # We should check if we can beat 30 with another n?
                        # Yes.
                    break # Stop trying this E for this n, move to next E or n?
                    # Actually, if we found one at E, we don't need to try larger E for this n.
                    # But we might find a smaller E with a different n?
                    # The outer loop is n. The inner is E.
                    # If we find a solution at n=10, E=30, we update best.
                    # We should continue checking other n's for potentially smaller E?
                    # Yes.
            
            # If we found a solution for this n, we can stop checking larger E for this n
            # because we want minimum E.
            if best_count <= E: 
                 # Wait, if we found one at E=30, best_count is 30.
                 # We don't need to try E=31 for this n.
                 # But we DO need to try other n's.
                 pass

    if best_edges:
        return {'n': n, 'edges': best_edges}
    else:
        # Fallback: The complete graph on 9 vertices (126 edges)
        # Or a known 51 edge construction if we had it.
        # Since we didn't find a < 51 one in random search (unlikely to fail completely if 51 is dense enough),
        # let's just return the 126 edge one as a safe fallback, 
        # BUT the random search for E up to 51 should find something.
        # Actually, for n=9, E=126 is the only one (complete).
        # For n=10, E=51 should be found easily.
        
        # If best_edges is still None, it means we failed to find any < 52.
        # This is highly unlikely.
        # Let's force a known structure if needed.
        # But for now, return whatever we found.
        
        # Fallback to n=9 complete
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

# EVOLVE-BLOCK-END
