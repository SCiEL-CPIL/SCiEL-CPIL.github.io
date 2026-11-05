import time
import random
from itertools import combinations

def construct():
    """
    Return a 5-uniform hypergraph with no proper 2-colouring, minimising edges.
    
    Strategy:
      1. Use a fast backtracking checker with unit propagation for n <= 26.
      2. Multi-start random search over n=13 to n=18.
      3. Aggressive greedy pruning.
      4. Safety: always return a valid non-2-colorable hypergraph.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit (evaluator kills at 150s)

    best_edges = None
    best_n = 14

    def is_non_2colorable_backtracking(edges, n):
        """
        Returns True if no 2-colouring exists where every edge has both colours.
        Uses backtracking with unit propagation.
        """
        if not edges:
            return False

        # Preprocess edges into lists of vertex indices
        edge_list = [list(e) for e in edges]
        
        # For unit propagation, we need to know which edges contain which vertices.
        # We can maintain a list of 'unresolved' edges.
        # State: assignment dict {vertex: color}, color in {0, 1}
        
        # To speed up, we can represent edges as sets or lists.
        # Unit propagation:
        # If an edge has k-1 vertices assigned the same color, the last one MUST be the other color.
        # If an edge has all vertices assigned the same color, contradiction (this coloring is invalid for Property B? 
        # Wait. Property B requires NO monochromatic edge.
        # So if an edge becomes monochromatic during propagation, the branch is INVALID (no solution exists for this branch).
        # Wait, the question is: does a valid 2-coloring EXIST?
        # If we find a contradiction (monochromatic edge), it means this partial assignment CANNOT be extended to a valid coloring.
        # So if ALL branches lead to contradictions, then NO valid coloring exists -> Non-2-colorable.
        
        # Let's implement a standard backtracking solver for "Is there a 2-coloring with no mono edges?".
        # If the solver finds a solution, return False (it is 2-colorable).
        # If the solver exhausts all options, return True (it is NOT 2-colorable).

        # Optimization: Order vertices by degree? Or just 0..n-1.
        # We will use a recursive function.
        
        # To make it fast, we can use bitmasks if n is small, but for n=20+ backtracking is better.
        
        # Let's use a simple recursive backtracking with memoization on (assigned_mask, assigned_colors)? 
        # State space is 3^n (0, 1, unassigned). Too big.
        # Standard backtracking with forward checking.
        
        # Let's try a specific algorithm:
        # 1. Pick an unassigned vertex.
        # 2. Try color 0.
        #    - Propagate: Check all edges containing this vertex.
        #      If an edge has 4 vertices color 0, the 5th MUST be 1. Assign it.
        #      If an edge has 5 vertices color 0, CONTRADICTION.
        #    - If no contradiction, recurse.
        # 3. Try color 1.
        #    - Propagate similarly.
        
        # We need efficient edge lookup.
        # vertex_to_edges: list of lists of edge indices.
        
        vertex_to_edges = [[] for _ in range(n)]
        for i, e in enumerate(edge_list):
            for v in e:
                vertex_to_edges[v].append(i)

        # State:
        # colors: list of length n, -1 for unassigned, 0/1 for assigned.
        # We can use a list and modify it in-place.
        
        # To speed up, we can pick the vertex with the most constraints (highest degree in unassigned graph?).
        # For simplicity, pick first unassigned.
        
        def propagate(colors, changed_vertex, color):
            """
            Perform unit propagation starting from changed_vertex with assigned color.
            Returns False if contradiction found.
            Modifies colors in-place.
            """
            stack = [changed_vertex]
            while stack:
                v = stack.pop()
                c = colors[v]
                for e_idx in vertex_to_edges[v]:
                    e = edge_list[e_idx]
                    # Check this edge
                    # Count colors in this edge
                    # We need to check if this edge triggers a unit clause or contradiction.
                    # Optimization: Only check if the edge contains v and v was just assigned?
                    # Actually, propagation might assign other vertices, so we should check edges of newly assigned vertices.
                    
                    # Let's just scan the edge.
                    # If all assigned vertices in e have the same color, and there is exactly 1 unassigned, assign it the other color.
                    # If all assigned vertices in e have the same color, and there are 0 unassigned, contradiction.
                    
                    assigned_colors = []
                    unassigned_verts = []
                    for u in e:
                        if colors[u] != -1:
                            assigned_colors.append(colors[u])
                        else:
                            unassigned_verts.append(u)
                    
                    if not assigned_colors:
                        continue
                        
                    # Check if all assigned are same
                    first_c = assigned_colors[0]
                    all_same = True
                    for ac in assigned_colors[1:]:
                        if ac != first_c:
                            all_same = False
                            break
                            
                    if all_same:
                        if len(unassigned_verts) == 0:
                            # Contradiction: monochromatic edge
                            return False
                        elif len(unassigned_verts) == 1:
                            # Unit propagation
                            new_v = unassigned_verts[0]
                            new_c = 1 - first_c
                            if colors[new_v] != -1:
                                # Already assigned?
                                if colors[new_v] != new_c:
                                    return False
                                else:
                                    continue
                            else:
                                colors[new_v] = new_c
                                stack.append(new_v)
            
            return True

        def backtrack(colors, unassigned_count):
            if unassigned_count == 0:
                # All assigned, no contradiction found so far
                # Check if any edge is mono? 
                # Propagation should have caught it.
                return True # Found a valid coloring

            # Heuristic: Pick unassigned vertex with highest degree in remaining graph?
            # For simplicity, pick first unassigned.
            # We can improve by picking vertex involved in most "tight" edges.
            # Let's just pick first unassigned for now to keep code simple and robust.
            
            # Find first unassigned
            v = -1
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
            
            if v == -1:
                return True

            # Try color 0
            colors[v] = 0
            if propagate(colors, v, 0):
                if backtrack(colors, unassigned_count - 1):
                    return True
            colors[v] = -1
            
            # Try color 1
            colors[v] = 1
            if propagate(colors, v, 1):
                if backtrack(colors, unassigned_count - 1):
                    return True
            colors[v] = -1
            
            return False

        colors = [-1] * n
        # Initial propagation? No, nothing assigned.
        # We can just call backtrack.
        # But we can also try to find a contradiction immediately if possible?
        # Backtrack will handle it.
        
        # Optimization: Symmetry breaking. 
        # If we fix vertex 0 to 0, we cover half the space. 
        # If a coloring exists with v0=1, the inverse coloring has v0=0.
        # So we can just fix v0=0 and search. If we find a solution, it's 2-colorable.
        # If we don't, is it non-2-colorable?
        # Yes, because if there was a solution with v0=1, its inverse would be a solution with v0=0.
        
        colors[0] = 0
        # Note: propagate might not be needed for initial assignment if we just set it, 
        # but it might trigger other assignments if there are edges with 4 unassigned? 
        # No, unit propagation only triggers if 4 are assigned same. 
        # So initial state is safe.
        
        # However, we must ensure that if v0=0 leads to no solution, we don't need to check v0=1.
        
        if backtrack(colors, n - 1):
            return False # 2-colorable
        else:
            return True # Not 2-colorable

    def prune_edges(edges, n, time_limit):
        """
        Greedy pruning with random restarts.
        """
        current_edges = list(edges)
        
        # Single pass
        current_edges = _prune_pass(current_edges, n, time_limit)
        
        # Random restarts
        max_restarts = 20
        for _ in range(max_restarts):
            if time.time() > time_limit:
                break
            if len(current_edges) <= 51:
                break
            
            shuffled = current_edges[:]
            random.shuffle(shuffled)
            
            new_edges = _prune_pass(shuffled, n, time_limit)
            
            if len(new_edges) < len(current_edges):
                current_edges = new_edges
            else:
                # Try a kick: remove a random edge, if fails, put back.
                if len(current_edges) > 1:
                    idx = random.randint(0, len(current_edges) - 1)
                    kick_edges = current_edges[:idx] + current_edges[idx+1:]
                    if is_non_2colorable_backtracking(kick_edges, n):
                        current_edges = kick_edges
                    else:
                        break # Stuck
                else:
                    break
                    
        return current_edges

    def _prune_pass(edges, n, time_limit):
        """
        Single pass of greedy pruning.
        """
        current_edges = list(edges)
        i = 0
        while i < len(current_edges):
            if time.time() > time_limit:
                break
            
            candidate = current_edges[:i] + current_edges[i+1:]
            
            if is_non_2colorable_backtracking(candidate, n):
                current_edges = candidate
            else:
                i += 1
        return current_edges

    # Pre-compute all possible edges for small n
    all_possible_edges = {}
    for n in [13, 14, 15, 16, 17, 18]:
        all_possible_edges[n] = list(combinations(range(n), 5))
        
    # Search loop
    for n in [13, 14, 15, 16, 17, 18]:
        if time.time() > deadline - 10.0:
            break
            
        total_edges = len(all_possible_edges[n])
        # Reduce attempts for larger n as checking is slower
        attempts = 200 if n <= 15 else 100

        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break
                
            # Start with a dense random graph
            # Density depends on n. 
            # For n=13, C(13,5)=1287. 
            # For n=18, C(18,5)=8568.
            # We want to start with enough edges to be non-2-colorable.
            # m(5) <= 51. 
            # Let's try to start with ~100 edges.
            
            k_init = 100
            
            all_possible = all_possible_edges[n][:]
            random.shuffle(all_possible)
            initial_edges = [list(e) for e in all_possible[:k_init]]
            
            if is_non_2colorable_backtracking(initial_edges, n):
                prune_deadline = time.time() + 2.0
                if time.time() + 2.0 > deadline:
                    prune_deadline = deadline - 1.0
                    
                pruned = prune_edges(initial_edges, n, prune_deadline)
                
                if best_edges is None or len(pruned) < len(best_edges):
                    best_edges = pruned
                    best_n = n
                    
                    if len(pruned) <= 51:
                        # Good result
                        pass

    # Fallback: Complete 5-graph on 9 vertices (126 edges)
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
