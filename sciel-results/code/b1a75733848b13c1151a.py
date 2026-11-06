import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable.
    Uses backtracking with unit propagation.
    edges: list of tuples of 5 vertices.
    n: number of vertices.
    """
    if not edges:
        return True
        
    # Pre-process edges for faster lookup
    # Map vertex to list of edge indices? Or just iterate.
    # For n<=10, m<=126, simple recursion is fine.
    
    # State: color of each vertex (-1 unknown, 0, 1)
    colors = [-1] * n
    
    # To speed up, we can pick the most constrained vertex?
    # Or just pick the first uncolored vertex.
    
    def backtrack():
        # Find first uncolored vertex
        v = -1
        for i in range(n):
            if colors[i] == -1:
                v = i
                break
        
        if v == -1:
            # All colored, check all edges
            for e in edges:
                c0 = 0
                c1 = 0
                for u in e:
                    if colors[u] == 0:
                        c0 += 1
                    else:
                        c1 += 1
                if c0 == 5 or c1 == 5:
                    return False
            return True
            
        # Try color 0
        colors[v] = 0
        # Check if this assignment forces any other vertex
        # Unit propagation:
        # If an edge has 4 vertices of color 0, the 5th must be 1.
        # If an edge has 5 vertices of color 0, contradiction.
        
        # Optimization: Check constraints for edges containing v
        # Actually, just check all edges for immediate contradiction?
        # Or propagate.
        
        # Simple propagation:
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                unknowns = []
                for u in e:
                    if colors[u] == 0:
                        c0 += 1
                    elif colors[u] == 1:
                        c1 += 1
                    else:
                        unknowns.append(u)
                
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and len(unknowns) == 1:
                    colors[unknowns[0]] = 1
                    changed = True
                elif c1 == 4 and len(unknowns) == 1:
                    colors[unknowns[0]] = 0
                    changed = True
                    
        # Recurse
        if backtrack():
            return True
            
        # Undo
        colors[v] = -1
        # Note: The propagation above might have set other colors.
        # We need to undo those too.
        # To keep it simple and safe, we can just not do in-place propagation in the recursion stack
        # or we store the changes.
        # Given n is small, we can just copy the colors array?
        # No, that's slow.
        # Let's use a simpler approach: no in-place propagation in the recursive step,
        # just check contradictions at the node.
        # Or, implement undo properly.
        
        # For now, let's revert any changes made by propagation?
        # The `while changed` loop modifies `colors`.
        # We need to track which vertices were set by propagation and revert them.
        
        # Revised Backtrack:
        # 1. Assign v
        # 2. Propagate
        # 3. Recurse
        # 4. Undo Propagation
        # 5. Unassign v
        
        # This is getting complex. Let's use a simpler solver for n<=10.
        # Brute force with pruning.
        
        # Actually, let's just use the simple check:
        # If we don't do propagation, we just branch on v.
        # The "unit propagation" is just an optimization.
        # For n=9, 2^9 = 512 states. We can just enumerate all 512 colorings?
        # Yes! 512 is tiny.
        
        return False

    # Since n is small (<= 10), we can just iterate through all 2^n colorings.
    # This is O(2^n * m).
    # For n=9, m=126: 512 * 126 = 64512 ops. Very fast.
    
    for mask in range(1 << n):
        valid = True
        for e in edges:
            c0 = 0
            c1 = 0
            for u in e:
                if (mask >> u) & 1:
                    c1 += 1
                else:
                    c0 += 1
            if c0 == 5 or c1 == 5:
                valid = False
                break
        if valid:
            return True
            
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 9
    # Start with complete 5-uniform hypergraph on 9 vertices
    all_edges = [tuple(e) for e in combinations(range(n), 5)]
    current_edges = all_edges[:]
    
    # Verify it's non-2-colorable
    if is_2_colorable(n, current_edges):
        # This should not happen for K9^5
        return {'n': n, 'edges': [list(e) for e in current_edges]}
        
    # Greedy deletion
    # We will try to remove edges one by one.
    # If removing an edge keeps it non-2-colorable, we remove it.
    
    # To make it faster, we shuffle and try to remove.
    # We do multiple passes.
    
    # Optimization: 
    # If we have time, we can do many passes.
    # Each pass: shuffle, iterate, try remove.
    
    # Since is_2_colorable is fast (brute force 2^n), we can do many checks.
    
    max_passes = 100
    for p in range(max_passes):
        if time.time() > deadline:
            break
            
        random.shuffle(current_edges)
        new_edges = []
        removed_any = False
        
        for edge in current_edges:
            if time.time() > deadline:
                new_edges.append(edge)
                continue
                
            # Try to remove edge
            # Create list without edge
            # To avoid O(m) copy, we can pass a flag? 
            # But is_2_colorable takes a list.
            # m is small (< 126). Copy is fine.
            
            candidate = [e for e in current_edges if e != edge]
            
            # Check if candidate is non-2-colorable
            if not is_2_colorable(n, candidate):
                # It is still non-2-colorable, so we can remove edge
                removed_any = True
                # Do not add to new_edges
            else:
                # It became 2-colorable, so we must keep edge
                new_edges.append(edge)
                
        current_edges = new_edges
        
        if not removed_any:
            # No more edges can be removed in this pass
            # It might be minimal, or we might need to change order?
            # Greedy deletion is order-dependent.
            # But we shuffle, so multiple passes help.
            # If no edge is removable, we are at a local minimum.
            # We can stop or try another shuffle.
            # Let's try a few more shuffles just in case.
            if p > 5:
                break
                
    return {'n': n, 'edges': [list(e) for e in current_edges]}
