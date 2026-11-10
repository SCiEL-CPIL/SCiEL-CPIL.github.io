import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110  # Leave 10s margin for 120s limit
    
    best_edges = None
    best_n = None
    min_edges_found = float('inf')
    
    # Solver
    def is_non_2_colorable(n, edges):
        # edges is a list of lists of 5 ints
        # Returns True if NOT 2-colorable
        
        # Preprocess: map edges to indices for faster access?
        # For n <= 26 and m <= 200, direct list access is fine.
        
        # Vertex to edges mapping
        v_to_e = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_to_e[v].append(i)
        
        colors = [-1] * n
        # counts for each edge: how many 0s, how many 1s
        c0 = [0] * len(edges)
        c1 = [0] * len(edges)
        
        # Propagation queue
        # We need to handle assignments and their consequences.
        # Since we are backtracking, we need to be able to undo.
        # Easier: use recursion with state copying?
        # State copying for n=26, m=200 is cheap.
        
        def propagate(colors, c0, c1, v, color):
            # Assign v to color
            if colors[v] != -1:
                if colors[v] != color:
                    return False
                else:
                    return True
            
            colors[v] = color
            if color == 0:
                c0_local = c0
            else:
                c0_local = c1 # careful, need separate arrays or pass both
                
            # Actually, let's just modify c0 and c1 in place and return a log of changes to undo?
            # Or simpler: just copy the arrays at the start of the recursive step.
            # With m=200, copying 2 arrays of 200 ints is fast.
            
            # Let's use a stack-based propagation inside the check.
            # But backtracking requires saving state.
            
            # Let's implement a standard DFS with state cloning.
            pass

        # Better approach: 
        # Use a specific backtracking function that takes the current state.
        
        def dfs(colors, c0, c1):
            # Check for conflicts
            # 1. Any edge with 5 of same color?
            # This is handled by propagation.
            
            # Find unassigned vertex with most constraints?
            # Or just first unassigned.
            # Let's try: pick vertex with smallest domain? 
            # Domain is 2 for all unassigned.
            # Heuristic: pick vertex involved in most "tight" edges (count 3 or 4).
            
            # For simplicity and speed, pick first unassigned.
            v = -1
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
            
            if v == -1:
                # All assigned, no conflict found during propagation
                return True # Wait, dfs returns "is it 2-colorable?"
                # If we reach here, we found a coloring.
                # So is_non_2_colorable should return False.
                # Let's define dfs returns True if a valid coloring EXISTS.
                return True
            
            # Try color 0
            new_colors = colors[:]
            new_c0 = c0[:]
            new_c1 = c1[:]
            
            if propagate_and_check(v, 0, new_colors, new_c0, new_c1, v_to_e, edges):
                if dfs(new_colors, new_c0, new_c1):
                    return True
            
            # Try color 1
            new_colors = colors[:]
            new_c0 = c0[:]
            new_c1 = c1[:]
            
            if propagate_and_check(v, 1, new_colors, new_c0, new_c1, v_to_e, edges):
                if dfs(new_colors, new_c0, new_c1):
                    return True
            
            return False
        
        def propagate_and_check(v, color, colors, c0, c1, v_to_e, edges):
            # Assign v to color
            if colors[v] != -1:
                return colors[v] == color
            
            colors[v] = color
            if color == 0:
                # Update counts for edges containing v
                for e_idx in v_to_e[v]:
                    c0[e_idx] += 1
                    if c0[e_idx] == 5:
                        return False
                    if c0[e_idx] == 4:
                        # Find the unassigned vertex in this edge
                        edge = edges[e_idx]
                        unassigned = -1
                        for u in edge:
                            if colors[u] == -1:
                                unassigned = u
                                break
                        if unassigned == -1:
                            # All assigned, but one must be 1?
                            # If 4 are 0, the 5th MUST be 1.
                            # If the 5th is already 0, we would have hit count 5.
                            # If the 5th is 1, count0 is 4, count1 is 1. OK.
                            # If the 5th is -1, we assign it 1.
                            pass
                        else:
                            # Force unassigned to 1
                            if not propagate_and_check(unassigned, 1, colors, c0, c1, v_to_e, edges):
                                return False
            else:
                for e_idx in v_to_e[v]:
                    c1[e_idx] += 1
                    if c1[e_idx] == 5:
                        return False
                    if c1[e_idx] == 4:
                        edge = edges[e_idx]
                        unassigned = -1
                        for u in edge:
                            if colors[u] == -1:
                                unassigned = u
                                break
                        if unassigned != -1:
                            if not propagate_and_check(unassigned, 0, colors, c0, c1, v_to_e, edges):
                                return False
            
            return True
        
        # Initial state
        init_colors = [-1] * n
        init_c0 = [0] * len(edges)
        init_c1 = [0] * len(edges)
        
        # Check if any edge is already monochromatic? No, all -1.
        
        if dfs(init_colors, init_c0, init_c1):
            return False # It IS 2-colorable
        else:
            return True # It is NOT 2-colorable

    # Minimization
    def minimize(n, edges):
        # edges is list of lists
        # Returns minimized list
        # We try to remove each edge.
        # To be efficient, we randomize the order.
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        
        current_edges = [list(e) for e in edges]
        
        # We iterate and try to remove.
        # Since removing an edge makes the hypergraph "easier" to color,
        # if it was non-2-colorable, it might become 2-colorable.
        
        # We need to check non-2-colorability after each removal.
        # This is expensive.
        # Optimization: Only check if the removed edge was "critical".
        # Hard to know without checking.
        
        # Just do it.
        # Limit iterations?
        
        i = 0
        while i < len(current_edges):
            if time.time() > deadline:
                break
                
            e_idx = edge_indices[i]
            if e_idx >= len(current_edges):
                i += 1
                continue
                
            # Try removing edge at e_idx
            # Note: current_edges indices shift if we remove.
            # Better: work with a set of edge tuples? Or just rebuild list.
            
            edge_to_remove = current_edges[e_idx]
            test_edges = [e for j, e in enumerate(current_edges) if j != e_idx]
            
            # Check if test_edges is non-2-colorable
            if is_non_2_colorable(n, test_edges):
                # It is still non-2-colorable, so we can remove it
                current_edges = test_edges
                # The indices in edge_indices are now invalid for the next steps if we just shifted.
                # Let's restart the minimization loop with the new list.
                # This is O(m^2) checks. With m=100, 100 checks.
                # Each check is fast?
                # Let's restart to keep indices consistent.
                return minimize(n, current_edges)
            else:
                # Cannot remove
                i += 1
                
        return current_edges

    # Main search
    # Try different n
    for n in range(9, 14):
        if time.time() > deadline:
            break
            
        # Generate a random non-2-colorable hypergraph
        # Start with random edges, add more if 2-colorable
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Random number of edges
            # Start with ~1.5 * best known?
            # Best known 51.
            # Let's start with 60 edges.
            m_start = 60
            # Ensure m_start <= total possible
            total_possible = len(list(combinations(range(n), 5)))
            if m_start > total_possible:
                m_start = total_possible
                
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            edges = [list(e) for e in all_edges[:m_start]]
            
            # Ensure non-2-colorable
            while is_non_2_colorable(n, edges) == False:
                # It is 2-colorable, add a random edge
                remaining = [e for e in all_edges if e not in edges]
                if not remaining:
                    break
                new_edge = random.choice(remaining)
                edges.append(list(new_edge))
                
            if is_non_2_colorable(n, edges):
                # Minimize
                min_edges = minimize(n, edges)
                if len(min_edges) < min_edges_found:
                    min_edges_found = len(min_edges)
                    best_edges = min_edges
                    best_n = n
                    
    if best_edges is None:
        # Fallback to seed
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
