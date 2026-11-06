import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    def build_coloring(n, max_restarts=5000):
        """
        Attempts to build a valid 4-coloring of K_n.
        Returns a list of colors for pairs (i, j) with i < j, or None if failed.
        """
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # adj[i][j] stores the color for edge (i, j). -1 means unassigned.
            adj = [[-1] * n for _ in range(n)]
            
            # We build vertex by vertex.
            # When adding vertex k, we assign colors to edges (k, 0), ..., (k, k-1).
            # We need to ensure no monochromatic triangle is formed.
            # A triangle (i, j, k) with i < j < k is monochromatic if adj[i][j] == adj[i][k] == adj[j][k].
            
            # To speed up, we can track for each color c, the set of vertices connected to k by color c.
            # However, a simpler check is: when assigning color c to (k, j),
            # we must ensure that for all i < j, adj[i][j] != c OR adj[i][k] != c.
            # Actually, if we assign (k, j) = c, we must check all pairs (i, j) where i < j < k.
            # If adj[i][j] == c, then adj[i][k] must NOT be c.
            
            # Let's use a backtracking search for the assignment of the last vertex's edges.
            # Or even simpler: Greedy with random restarts.
            
            def is_valid_assignment(k, colors_for_k):
                # colors_for_k[j] is the color for edge (k, j) for j in 0..k-1
                # Check all triangles involving k.
                # Triangles are (i, j, k) with i < j < k.
                # Edge (i, j) is adj[i][j].
                # Edge (i, k) is colors_for_k[i].
                # Edge (j, k) is colors_for_k[j].
                # Condition: NOT (adj[i][j] == colors_for_k[i] == colors_for_k[j])
                
                for i in range(k):
                    c_i = colors_for_k[i]
                    for j in range(i + 1, k):
                        c_j = colors_for_k[j]
                        if c_i == c_j and adj[i][j] == c_i:
                            return False
                return True

            # Greedy approach:
            # For k from 0 to n-1:
            #   If k == 0, no edges.
            #   If k > 0, assign colors to (k, 0)...(k, k-1).
            #   We can pick colors randomly but check validity.
            #   To make it smarter, we can pick colors that minimize future conflicts,
            #   but for now, random with backtracking on the last few vertices might be enough.
            
            # Let's try a simpler heuristic:
            # For each new vertex k, try to assign colors to its edges (k, 0)...(k, k-1)
            # such that no monochromatic triangle is formed.
            # We can do this by picking colors one by one for j=0..k-1.
            # When picking color for (k, j), we know colors for (k, 0)..(k, j-1).
            # We must ensure that for any i < j, if adj[i][j] == c_candidate, then colors_for_k[i] != c_candidate.
            
            def assign_vertex(k):
                # Returns list of colors for edges (k, 0)...(k, k-1) or None
                if k == 0:
                    return []
                
                # Backtracking search for the k-1 edges
                colors = [None] * k
                
                def backtrack(j):
                    if j == k:
                        return True
                    
                    # Try colors 0-3 in random order
                    cs = [0, 1, 2, 3]
                    random.shuffle(cs)
                    
                    for c in cs:
                        colors[j] = c
                        
                        # Check if adding (k, j) with color c creates a monochromatic triangle
                        # with any i < j.
                        # Triangle (i, j, k) is monochromatic if adj[i][j] == c and colors[i] == c.
                        # Note: colors[i] is already assigned because i < j and we fill 0..k-1.
                        
                        valid = True
                        for i in range(j):
                            if adj[i][j] == c and colors[i] == c:
                                valid = False
                                break
                                
                        if valid:
                            if backtrack(j + 1):
                                return True
                                
                        colors[j] = None
                    return False

                if backtrack(0):
                    return colors
                return None

            # Build the graph
            success = True
            for k in range(n):
                if k == 0:
                    continue
                    
                if time.time() > deadline:
                    return None
                    
                cols = assign_vertex(k)
                if cols is None:
                    success = False
                    break
                
                # Commit the colors
                for j in range(k):
                    adj[k][j] = cols[j]
                    adj[j][k] = cols[j]
                    
            if success:
                # Format output: flat list of colors for (i, j) with i < j
                res = []
                for i in range(n):
                    for j in range(i + 1, n):
                        res.append(adj[i][j])
                return res
        return None

    # Try to build for n=50 (Chung's record)
    # If it fails, try smaller n
    for target_n in [50, 45, 40, 35, 32]:
        if time.time() > deadline - 5: # Leave some time for formatting
            break
            
        colors = build_coloring(target_n, max_restarts=1000)
        if colors is not None:
            return {'n': target_n, 'colors': colors}
            
    # Fallback: Return a trivial valid coloring for small n
    # n=1 is trivial, n=2 is trivial.
    # Let's return n=1 to be safe, though n=32 should have been found.
    # Actually, let's just return n=1 if all else fails.
    return {'n': 1, 'colors': []}
