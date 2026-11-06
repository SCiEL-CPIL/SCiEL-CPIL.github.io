import time
import sys

def construct():
    start_time = time.time()
    time_limit = 110
    
    # We aim for n=26, then 25, etc.
    # The Lazebnik-Woldar construction gives 26.
    # We will try to find a valid coloring for n=26 using backtracking.
    
    def solve(n):
        # State:
        # adj[c][i] is a bitmask of neighbors of i in color c.
        adj = [[0] * n for _ in range(5)]
        
        # Precompute edge indices in lexicographic order
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                
        # Recursive backtracking
        # To avoid recursion depth issues and overhead, we might use iterative or optimized recursion.
        # Python recursion limit is ~1000, so depth 325 is fine.
        
        def backtrack(edge_idx):
            if time.time() - start_time > time_limit:
                return False
            
            if edge_idx == len(edges):
                return True
            
            i, j = edges[edge_idx]
            
            # Try each color
            for c in range(5):
                # Check if adding edge (i, j) with color c is valid
                
                # 1. Check common neighbors of i and j in color c
                # N_i = adj[c][i], N_j = adj[c][j]
                # Common = N_i & N_j
                # If popcount(Common) >= 2, invalid.
                common_ij = adj[c][i] & adj[c][j]
                if bin(common_ij).count('1') >= 2:
                    continue
                
                # 2. Check impact on other pairs
                # For any k != i, j:
                # If j is in adj[c][k], then i and k must have 0 common neighbors (before adding? No, we check the new state).
                # Let's check the condition:
                # New N_i includes j.
                # For pair (i, k):
                #   If j in N_k, then j is a common neighbor of i and k.
                #   Existing common neighbors of i and k: N_i & N_k.
                #   Total common neighbors = (N_i & N_k) + (1 if j in N_k else 0).
                #   Note: j is not in N_i (before adding), so j is not in N_i & N_k.
                #   So we need: if j in N_k, then (N_i & N_k) must be 0.
                
                # Similarly for pair (j, k):
                #   If i in N_k, then (N_j & N_k) must be 0.
                
                # We can combine these checks.
                # Iterate k from 0 to n-1, k != i, j.
                # This is O(n) per color attempt.
                # n=26, so 26 checks.
                # Each check involves bitwise AND and popcount.
                
                valid = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    
                    # Check pair (i, k)
                    if (adj[c][k] >> j) & 1: # j is neighbor of k
                        # Then i and k must have no common neighbors in adj[c][i] & adj[c][k]
                        if (adj[c][i] & adj[c][k]) != 0:
                            valid = False
                            break
                    
                    # Check pair (j, k)
                    if (adj[c][k] >> i) & 1: # i is neighbor of k
                        # Then j and k must have no common neighbors in adj[c][j] & adj[c][k]
                        if (adj[c][j] & adj[c][k]) != 0:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Also, we must ensure that adding (i, j) doesn't violate the condition for (i, j) itself?
                # We checked common_ij < 2.
                # But wait, does adding (i, j) change the common neighbors of (i, j)?
                # Yes, i and j become neighbors.
                # But the definition is "two vertices have two common neighbors".
                # i and j are the two vertices. Their common neighbors are N_i & N_j.
                # We checked that.
                
                # What about other pairs?
                # We checked (i, k) and (j, k).
                # Are there any other pairs affected?
                # Only pairs involving i or j have their neighbor sets changed.
                # So this is sufficient.
                
                # If valid, make the move
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
                if backtrack(edge_idx + 1):
                    return True
                
                # Undo move
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                
            return False
        
        # Optimization: Symmetry breaking
        # Fix the first few edges to specific colors to reduce search space.
        # Edge (0, 1) -> Color 0
        # Edge (0, 2) -> Color 0 or 1?
        # Let's just fix (0,1) to 0.
        
        # We can also order edges to fail fast.
        # Lexicographic is okay.
        
        if not backtrack(0):
            return None
        
        # Extract colors
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                # Find color c such that bit j is set in adj[c][i]
                c_found = -1
                for c in range(5):
                    if (adj[c][i] >> j) & 1:
                        c_found = c
                        break
                colors.append(c_found)
        
        return {'n': n, 'colors': colors}

    # Try n=26
    if time.time() - start_time < time_limit:
        res = solve(26)
        if res:
            return res

    # Try n=25
    if time.time() - start_time < time_limit:
        res = solve(25)
        if res:
            return res

    # Try n=24
    if time.time() - start_time < time_limit:
        res = solve(24)
        if res:
            return res

    # Fallback
    return {'n': 1, 'colors': []}
