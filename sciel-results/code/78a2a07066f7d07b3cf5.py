import time
import itertools
import random

DEADLINE = 140.0

def _unit_propagate(edges, colors):
    """
    Returns (new_colors, is_valid)
    is_valid is False if a conflict is found.
    """
    n = len(colors)
    changed = True
    while changed:
        changed = False
        for edge in edges:
            c0 = 0
            c1 = 0
            uncolored = []
            for v in edge:
                if colors[v] == 0:
                    c0 += 1
                elif colors[v] == 1:
                    c1 += 1
                else:
                    uncolored.append(v)
            
            if c0 == 5 or c1 == 5:
                return colors, False
            
            if len(uncolored) == 1:
                if c0 == 4:
                    colors[uncolored[0]] = 1
                    changed = True
                elif c1 == 4:
                    colors[uncolored[0]] = 0
                    changed = True
    return colors, True

def _is_not_2_colorable(edges, n):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    """
    colors = [-1] * n
    
    # Heuristic: order vertices by degree (number of edges they appear in)
    # This helps backtracking prune faster.
    degrees = [0] * n
    for edge in edges:
        for v in edge:
            degrees[v] += 1
    
    # We don't strictly need to reorder the vertex list in the array, 
    # but we can pick the next variable based on degree among uncolored ones.
    
    def backtrack(idx):
        # Find next uncolored vertex with highest degree
        next_v = -1
        max_deg = -1
        for v in range(n):
            if colors[v] == -1 and degrees[v] > max_deg:
                max_deg = degrees[v]
                next_v = v
        
        if next_v == -1:
            # All colored. Check for conflicts (should have been caught by propagation)
            # But just in case:
            for edge in edges:
                if all(colors[v] == 0 for v in edge) or all(colors[v] == 1 for v in edge):
                    return False
            return True # Valid coloring found, so it IS 2-colorable. 
                        # Wait, the function name is _is_not_2_colorable.
                        # If we find a valid coloring, it IS 2-colorable, so return False.
        
        # Try color 0
        old_c = colors[next_v]
        colors[next_v] = 0
        new_colors, valid = _unit_propagate(edges, colors[:])
        if valid:
            # Save state? No, we modify in place and restore.
            # But _unit_propagate returns a copy in my previous thought. 
            # Let's make _unit_propagate modify in place and return bool.
            pass 
            
        # Let's rewrite _unit_propagate to modify in place and return success/failure.
        
        # This structure is getting messy. Let's use a cleaner recursive structure.
        pass
        
    # Simpler approach:
    # Use a stack or recursion with explicit state copying if n is small (<=26).
    # Copying a list of 26 ints is cheap.
    
    def bt(colors):
        # Check if all colored
        uncolored = [i for i in range(n) if colors[i] == -1]
        if not uncolored:
            # Valid coloring found
            return True 
            
        # Pick variable: uncolored vertex in an edge with most constraints?
        # Or just highest degree.
        v = max(uncolored, key=lambda x: degrees[x])
        
        for c in [0, 1]:
            new_colors = colors[:]
            new_colors[v] = c
            propagated, valid = _unit_propagate(edges, new_colors)
            if valid:
                if bt(propagated):
                    return True
        return False
        
    # If bt returns True, a coloring exists -> IS 2-colorable -> return False
    # If bt returns False, no coloring exists -> NOT 2-colorable -> return True
    return not bt(colors)

def construct():
    start_time = time.time()
    best_result = None
    best_e = 126 # Fallback: complete hypergraph on 9 vertices
    
    # Fallback construction
    fallback_edges = [list(e) for e in itertools.combinations(range(9), 5)]
    
    # Search for better (fewer edges) non-2-colorable hypergraphs
    # Try different n and e
    # n=15 is a good candidate. C(15,5) = 3003.
    # n=16: C(16,5) = 4368.
    # n=20: C(20,5) = 15504.
    
    rng = random.Random(42)
    
    # We want to find a hypergraph with e < best_e that is NOT 2-colorable.
    # Start with small e and increase.
    # If e is too small, it's likely 2-colorable.
    # The gap is 29 to 51. Let's try e in [30, 51].
    
    for n in [15, 16, 18, 20]:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        all_possible = list(itertools.combinations(range(n), 5))
        max_e_possible = len(all_possible)
        
        # Try e from 30 up to 51
        # If we find one with e=30, we stop searching for smaller e (impossible) 
        # but we might find one with e=29? The prompt says <=28 is always 2-colorable.
        # So min possible is 29.
        
        for e in range(29, 52):
            if time.time() - start_time > DEADLINE - 5:
                break
            if e >= best_e:
                break # We already have a better one (or equal), and we are increasing e
            
            # Try a few random samples for this (n, e)
            # Probability of being non-2-colorable increases with e.
            # For n=15, e=30, it might be rare. For e=50, it might be common.
            # Let's try 20 samples per (n, e) pair.
            
            found = False
            for _ in range(20):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                edges = rng.sample(all_possible, e)
                edges_list = [list(edge) for edge in edges]
                
                # Check if not 2-colorable
                if _is_not_2_colorable(edges_list, n):
                    # Found a valid one with e edges
                    best_e = e
                    best_result = {'n': n, 'edges': edges_list}
                    found = True
                    break
            
            if found:
                # Since we are iterating e upwards, the first found is the smallest for this n.
                # But we might find a smaller e for a different n?
                # Unlikely, as smaller n usually requires higher density to be non-2-colorable.
                # But let's continue to check if we can find e < best_e in other n's?
                # No, we are iterating e upwards. If we found e=35, we break the inner loop.
                # Then we go to next n. If next n finds e=32, that's better.
                # So we should not break the outer loop immediately unless we hit e=29.
                if best_e == 29:
                    break
                else:
                    continue
            
            if best_e == 29:
                break

    if best_result is not None:
        return best_result
    else:
        return {'n': 9, 'edges': fallback_edges}

# The _unit_propagate and _is_not_2_colorable functions need to be defined correctly.
# Let's re-verify the logic.

def _unit_propagate(edges, colors):
    """
    colors: list of ints (-1, 0, 1)
    Returns (new_colors, valid)
    """
    # We copy the list to avoid side effects in the caller if we don't want to modify in place?
    # The caller in bt() does new_colors = colors[:] and passes that.
    # So we can modify colors in place here.
    changed = True
    while changed:
        changed = False
        for edge in edges:
            c0 = 0
            c1 = 0
            uncolored = []
            for v in edge:
                if colors[v] == 0:
                    c0 += 1
                elif colors[v] == 1:
                    c1 += 1
                else:
                    uncolored.append(v)
            
            if c0 == 5 or c1 == 5:
                return colors, False
            
            if len(uncolored) == 1:
                if c0 == 4:
                    colors[uncolored[0]] = 1
                    changed = True
                elif c1 == 4:
                    colors[uncolored[0]] = 0
                    changed = True
    return colors, True

def _is_not_2_colorable(edges, n):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    """
    # Calculate degrees
    degrees = [0] * n
    for edge in edges:
        for v in edge:
            degrees[v] += 1
    
    def bt(colors):
        # Find uncolored vertex with max degree
        uncolored = [i for i in range(n) if colors[i] == -1]
        if not uncolored:
            return True # Found a valid coloring -> IS 2-colorable
        
        v = max(uncolored, key=lambda x: degrees[x])
        
        for c in [0, 1]:
            new_colors = colors[:]
            new_colors[v] = c
            propagated, valid = _unit_propagate(edges, new_colors)
            if valid:
                if bt(propagated):
                    return True
        return False
        
    initial_colors = [-1] * n
    # If bt returns True, a coloring exists.
    # We want to know if it is NOT 2-colorable.
    return not bt(initial_colors)
