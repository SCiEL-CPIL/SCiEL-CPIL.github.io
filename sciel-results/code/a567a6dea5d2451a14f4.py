import itertools
import random

def is_2_colorable(n, edges):
    # edges: list of tuples of 5 vertices
    if not edges:
        return True
    
    # Use a list for colors: -1 unassigned, 0/1 assigned
    # We need to be careful with state restoration in backtracking.
    
    def propagate(colors):
        # Returns (success, new_colors)
        # new_colors is a copy if changed, or same ref if not? 
        # For simplicity and safety, always copy if we modify.
        # But copying is expensive. 
        # Let's use a stack of changes? 
        # Or just copy the list. n is small (<=26). 
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0, c1, c2, c3, c4 = colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]], colors[e[4]]
                assigned = [c for c in (c0, c1, c2, c3, c4) if c != -1]
                unassigned_idx = [i for i, c in enumerate((c0, c1, c2, c3, c4)) if c == -1]
                
                if len(assigned) == 5:
                    if len(set(assigned)) == 1:
                        return False, colors
                elif len(assigned) == 4:
                    if len(set(assigned)) == 1:
                        val = 1 - assigned[0]
                        idx = unassigned_idx[0]
                        v = e[idx]
                        if colors[v] == -1:
                            colors[v] = val
                            changed = True
                        elif colors[v] != val:
                            return False, colors
        return True, colors

    def backtrack(colors, v_idx):
        if v_idx == n:
            return True
        
        v = v_idx
        for c in (0, 1):
            new_colors = colors[:]
            new_colors[v] = c
            success, new_colors = propagate(new_colors)
            if success:
                if backtrack(new_colors, v_idx + 1):
                    return True
        return False

    colors = [-1] * n
    success, colors = propagate(colors)
    if not success:
        return False
    return backtrack(colors, 0)

def construct():
    # Aim for n=10 or 11
    # We will try to find a non-2-colorable hypergraph with minimal edges.
    
    best_edges = None
    best_len = 1000
    
    # Try n=10
    n = 10
    all_combos = list(itertools.combinations(range(n), 5))
    
    for restart in range(100):
        # Start with a random set of edges
        # Size heuristic: maybe 30-50 edges
        k = 40
        current_edges = set(random.sample(all_combos, k))
        
        # If it's already non-2-colorable, great.
        # If it is 2-colorable, add random edges until it's not.
        if is_2_colorable(n, list(current_edges)):
            # Add edges
            remaining = [e for e in all_combos if e not in current_edges]
            random.shuffle(remaining)
            for e in remaining:
                current_edges.add(e)
                if not is_2_colorable(n, list(current_edges)):
                    break
            
            if is_2_colorable(n, list(current_edges)):
                continue # Failed to make it non-2-colorable?
        
        # Minimize
        edges_list = list(current_edges)
        random.shuffle(edges_list)
        working_edges = set(edges_list)
        
        for e in edges_list:
            if e not in working_edges:
                continue
            candidate = working_edges - {e}
            if not is_2_colorable(n, list(candidate)):
                working_edges.remove(e)
        
        if len(working_edges) < best_len:
            best_len = len(working_edges)
            best_edges = list(working_edges)
            
            if best_len <= 51:
                break # Can't get much better than known bound
                
    if best_edges is None:
        # Fallback: try n=11
        n = 11
        all_combos = list(itertools.combinations(range(n), 5))
        for restart in range(50):
            k = 50
            current_edges = set(random.sample(all_combos, k))
            if is_2_colorable(n, list(current_edges)):
                remaining = [e for e in all_combos if e not in current_edges]
                random.shuffle(remaining)
                for e in remaining:
                    current_edges.add(e)
                    if not is_2_colorable(n, list(current_edges)):
                        break
                if is_2_colorable(n, list(current_edges)):
                    continue
            
            edges_list = list(current_edges)
            random.shuffle(edges_list)
            working_edges = set(edges_list)
            for e in edges_list:
                if e not in working_edges:
                    continue
                candidate = working_edges - {e}
                if not is_2_colorable(n, list(candidate)):
                    working_edges.remove(e)
            
            if len(working_edges) < best_len:
                best_len = len(working_edges)
                best_edges = list(working_edges)
                if best_len <= 51:
                    break

    return {'n': n, 'edges': best_edges}
