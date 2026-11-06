import time
import itertools
import random

def _is_2_colorable(n, edges):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    # Precompute edge membership for faster access
    # edges is a list of tuples of vertices
    # We use a dictionary mapping edge index to the set of vertices
    # But for backtracking, we need to track constraints.
    
    # Simplified backtracking:
    # State: assignment of vertices to 0 or 1, or -1 for unassigned.
    # We use a stack for backtracking.
    
    # To make it fast, we use a simple recursive backtracker with memoization or just pruning.
    # Given n <= 26 and edges <= 200, a simple backtracker might be slow if not optimized.
    # However, for n=9, it is very fast.
    
    # Unit Propagation:
    # If an edge has 4 vertices assigned to 0, the 5th MUST be 1.
    # If an edge has 5 vertices assigned to 0, it's a conflict.
    
    def backtrack(assignment, edge_constraints):
        # edge_constraints: list of (edge, count_0, count_1, unassigned_vertices_in_edge)
        # This is complex to maintain. 
        # Simpler: Just check all edges at each step? 
        # For n=9, edges=126, depth=9. 126*2^9 is small.
        
        # Find unassigned vertex
        next_v = -1
        for v in range(n):
            if assignment[v] == -1:
                next_v = v
                break
        
        if next_v == -1:
            # All assigned. Check if any edge is monochromatic.
            for e in edges:
                colors = [assignment[v] for v in e]
                if all(c == colors[0] for c in colors):
                    return False # Conflict found, so this path is invalid for 2-coloring?
                    # Wait. If we found a monochromatic edge, it means this assignment is NOT a valid 2-coloring.
                    # We want to know if there EXISTS a valid 2-coloring.
                    # So if we reach the end and NO edge is monochromatic, it IS 2-colorable.
            return True

        # Try 0
        assignment[next_v] = 0
        if backtrack(assignment, edge_constraints):
            return True
        
        # Try 1
        assignment[next_v] = 1
        if backtrack(assignment, edge_constraints):
            return True
        
        assignment[next_v] = -1
        return False

    # Optimization: Prune early.
    # If an edge becomes monochromatic during assignment, we can prune.
    
    def backtrack_opt(assignment):
        # Check for conflicts (monochromatic edges)
        for e in edges:
            c0 = 0
            c1 = 0
            unassigned = 0
            for v in e:
                if assignment[v] == 0:
                    c0 += 1
                elif assignment[v] == 1:
                    c1 += 1
                else:
                    unassigned += 1
            if c0 == 5:
                return False # Conflict
            if c1 == 5:
                return False # Conflict
            # Unit propagation: if c0 == 4, the remaining must be 1.
            # We can implement this, but for n=9, simple backtracking is fast enough.
            
        # Find next unassigned
        next_v = -1
        for v in range(n):
            if assignment[v] == -1:
                next_v = v
                break
                
        if next_v == -1:
            return True # All assigned, no conflicts found in the loop above
            
        # Heuristic: Pick vertex with highest degree? Or just first.
        # For n=9, first is fine.
        
        assignment[next_v] = 0
        if backtrack_opt(assignment):
            return True
            
        assignment[next_v] = 1
        if backtrack_opt(assignment):
            return True
            
        assignment[next_v] = -1
        return False

    assignment = [-1] * n
    return backtrack_opt(assignment)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n = 9
    # Generate all 5-subsets of 0..8
    all_edges = [tuple(e) for e in itertools.combinations(range(n), 5)]
    
    # Start with all edges
    current_edges = list(all_edges)
    
    # Greedy removal:
    # Try to remove each edge. If the remaining graph is still non-2-colorable, remove it.
    # We iterate multiple times until no more edges can be removed or time runs out.
    
    # To make it deterministic and potentially find a smaller set, we can shuffle the order?
    # No, deterministic is better for reproducibility, but random might find a better local minimum.
    # Let's use a fixed order first.
    
    removed = True
    while removed and time.time() < deadline:
        removed = False
        # Iterate backwards to allow removal from the list while iterating
        for i in range(len(current_edges) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing edge i
            candidate_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if candidate is non-2-colorable
            # _is_2_colorable returns True if it IS 2-colorable.
            # We want it to be NOT 2-colorable, so _is_2_colorable should return False.
            if not _is_2_colorable(n, candidate_edges):
                # It is still non-2-colorable, so we can remove the edge
                current_edges = candidate_edges
                removed = True
    
    # Convert to list of lists for output
    edges_out = [list(e) for e in current_edges]
    
    return {'n': n, 'edges': edges_out}
