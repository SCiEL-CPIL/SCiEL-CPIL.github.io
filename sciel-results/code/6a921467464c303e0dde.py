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
        # Fast DPLL solver
        m = len(edges)
        if m == 0:
            return False
            
        # Precompute adjacency
        v_to_e = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                v_to_e[v].append(i)
        
        # State
        # colors: -1 unassigned, 0, 1
        colors = [-1] * n
        # For each edge, count of 0s and 1s
        c0 = [0] * m
        c1 = [0] * m
        
        # Trail for undo
        trail = [] # list of (vertex, old_color)
        # We also need to undo c0/c1 changes?
        # Easier: just copy c0/c1 if we go deep?
        # Or use a "stack" of edge count changes.
        # Given n<=26, m<=200, copying c0/c1 (200 ints) is very fast in Python if using lists?
        # Actually, list slicing `c0[:]` is O(m).
        # Recursion depth n. Total cost O(n*m). 26*200 = 5200 ops per node.
        # Branching factor 2. 2^26 is too big.
        # But propagation prunes heavily.
        
        # Let's stick to the copy-based method but optimize the inner loop.
        # The previous implementation was recursive with copies.
        
        def dfs(colors, c0, c1):
            # Check for unsat
            # If any c0[e] == 5 or c1[e] == 5, return False (Unsat)
            # This is handled by propagation.
            
            # Find unassigned vertex
            # Heuristic: pick vertex in an edge with max (c0+c1) or min domain?
            # Domain is always 2.
            # Pick vertex with highest "degree" in terms of tight edges?
            # Simple: first unassigned.
            v = -1
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
                    
            if v == -1:
                # All assigned
                # Check if any edge is monochromatic?
                # If we reached here, propagation didn't kill it.
                # But we must ensure no edge is mono.
                # Propagation ensures that if 4 are same, 5th is forced to other.
                # So if all assigned, no edge can have 5 same?
                # Wait. If an edge has 5 same, c0 or c1 would be 5.
                # Propagation returns False if c0==5 or c1==5.
                # So if we reach here, it is a valid coloring.
                return True
            
            # Try 0
            new_c0 = c0[:]
            new_c1 = c1[:]
            new_colors = colors[:]
            
            # Assign v=0
            new_colors[v] = 0
            ok = True
            for e_idx in v_to_e[v]:
                new_c0[e_idx] += 1
                if new_c0[e_idx] == 5:
                    ok = False
                    break
                if new_c0[e_idx] == 4:
                    # Force the other 4 to be 1? No, force the remaining one to be 1.
                    edge = edges[e_idx]
                    # Find the unassigned vertex in this edge
                    # We know v is now 0.
                    # c0 is 4. So 4 vertices in this edge are 0.
                    # The 5th must be 1.
                    # Who is the 5th?
                    # It's the one that is not 0.
                    # Since only v was just assigned, and others were either assigned or unassigned.
                    # If the 5th was already assigned 0, c0 would be 5 (caught above).
                    # If the 5th was assigned 1, c1 would be 1, c0 would be 4. OK.
                    # If the 5th was unassigned, we must assign it 1.
                    
                    # Identify the unassigned vertex in edge
                    unassigned_v = -1
                    for u in edge:
                        if new_colors[u] == -1:
                            unassigned_v = u
                            break
                            
                    if unassigned_v != -1:
                        # Recursive assignment? 
                        # This breaks the simple "try 0 then try 1" structure if we do deep recursion here.
                        # We should handle propagation in a loop before branching.
                        pass
            # This inline propagation is messy.
            
            # Let's use a helper that propagates and returns False if conflict.
            # It modifies colors, c0, c1 in place.
            # We pass copies.
            
            def propagate(colors, c0, c1, v, color):
                if colors[v] != -1:
                    return colors[v] == color
                colors[v] = color
                if color == 0:
                    for e_idx in v_to_e[v]:
                        c0[e_idx] += 1
                        if c0[e_idx] == 5:
                            return False
                        if c0[e_idx] == 4:
                            edge = edges[e_idx]
                            unassigned_v = -1
                            for u in edge:
                                if colors[u] == -1:
                                    unassigned_v = u
                                    break
                            if unassigned_v != -1:
                                if not propagate(colors, c0, c1, unassigned_v, 1):
                                    return False
                else:
                    for e_idx in v_to_e[v]:
                        c1[e_idx] += 1
                        if c1[e_idx] == 5:
                            return False
                        if c1[e_idx] == 4:
                            edge = edges[e_idx]
                            unassigned_v = -1
                            for u in edge:
                                if colors[u] == -1:
                                    unassigned_v = u
                                    break
                            if unassigned_v != -1:
                                if not propagate(colors, c0, c1, unassigned_v, 0):
                                    return False
                return True

            # Try 0
            if propagate(new_colors, new_c0, new_c1, v, 0):
                if dfs(new_colors, new_c0, new_c1):
                    return True
            
            # Try 1
            new_colors2 = colors[:]
            new_c02 = c0[:]
            new_c12 = c1[:]
            if propagate(new_colors2, new_c02, new_c12, v, 1):
                if dfs(new_colors2, new_c02, new_c12):
                    return True
            
            return False

        init_colors = [-1] * n
        init_c0 = [0] * m
        init_c1 = [0] * m
        
        if dfs(init_colors, init_c0, init_c1):
            return False
        else:
            return True

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
        # Start with m = 30, increase if 2-colorable
        # We want to find the minimum, so we should search low m first?
        # But low m is likely 2-colorable.
        # Let's jump to m=40, 50.
        
        # Better: Randomly generate, check, if non-2-colorable, minimize.
        # Keep the best minimized result.
        
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
