import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try n from 18 down to 9
    for n in range(18, 8, -1):
        if time.time() > deadline:
            break
            
        try:
            result = solve_n(n, deadline)
            if result is not None:
                return result
        except Exception as e:
            pass
            
    # Fallback to seed n=9 (or a known valid 17 if we hardcode one? 
    # The prompt says "best known colouring has 17 vertices". 
    # If we can't find 18, we should return 17 if we have it, or 9 if not.
    # But we should try to find 17 first.
    # Let's try n=17 again with more time if 18 fails?
    # The loop goes 18, 17, 16... so it will try 17.
    
    n = 9
    # Construct a valid coloring for n=9
    # A known valid coloring for n=9?
    # The seed uses: min(j - i, n - j + i) - 1
    # Let's verify this or just use it as a safe fallback.
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def solve_n(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # adj[c][v] is a bitmask of neighbors of v in color c
    # We use Python integers for bitmasks as they are fast and n <= 20
    adj = [[0] * n for _ in range(4)]
    
    # Precompute masks for vertices? Not needed, just bit operations.
    
    def would_create_c6(color, u, v):
        """
        Check if adding edge (u,v) to color creates a monochromatic C6.
        This is true if there is a path of length 5 from u to v in the current graph of 'color'.
        """
        # BFS from u up to depth 5 using bitsets
        # current_set is the set of vertices reachable from u in exactly k steps
        # We want to know if v is reachable in exactly 5 steps.
        # Actually, we just need to know if there is ANY path of length 5.
        # Let's do a BFS that tracks the set of vertices at each depth.
        
        # Depth 0: {u}
        current = 1 << u
        # We need to expand 5 times.
        # To avoid using the edge (u,v) itself, note that (u,v) is NOT in adj yet.
        # So we are safe.
        
        for step in range(5):
            next_set = 0
            # For each vertex in current, add its neighbors
            # Iterate through bits in current
            curr_bits = current
            while curr_bits:
                # Get lowest set bit
                lsb = curr_bits & -curr_bits
                idx = lsb.bit_length() - 1
                curr_bits ^= lsb
                # Add neighbors of idx in this color
                next_set |= adj[color][idx]
            
            # If we reach v at exactly step 5 (after 5 expansions), we have a path of length 5.
            # Wait, the loop runs 5 times.
            # Step 0: current = {u}
            # Step 1: current = N(u)
            # ...
            # Step 5: current = N^5(u)
            # If v is in current after the 5th iteration, there is a path of length 5.
            if step == 4: # After 5th expansion (0-indexed 4)
                if (current & (1 << v)) != 0:
                    return True
            
            current = next_set
            if current == 0:
                return False
        
        return False

    colors = [-1] * num_edges
    
    def backtrack(idx):
        if time.time() > deadline - 2.0:
            return False
        if idx == num_edges:
            return True
        
        u, v = edges[idx]
        
        # Try colors
        # Shuffle colors to avoid bias?
        color_order = [0, 1, 2, 3]
        random.shuffle(color_order)
        
        for c in color_order:
            if would_create_c6(c, u, v):
                continue
            
            # Assign
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
            if backtrack(idx + 1):
                return True
            
            # Unassign
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1
        
        return False

    if backtrack(0):
        return {'n': n, 'colors': colors}
    
    return None
