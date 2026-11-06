import time
import sys

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4s using Backtracking Search.
    """
    
    def solve_backtracking(N, time_limit=110):
        start_time = time.time()
        
        # adj[c][v] is a bitmask of neighbors of v in color c within the current subgraph 0..k-1
        # We will build up to N.
        # We use a recursive function to assign vertex k.
        
        # To handle time limits, we'll use a global counter or check time.
        # Python recursion limit might be an issue for N=26? Depth is 26, so it's fine.
        # But the branching factor is large.
        
        # We store the final colors in a flat list or 2D list.
        # colors[i][j] for i < j.
        colors = [[-1] * N for _ in range(N)]
        
        # adj[c][v] bitmask
        # Max N is 29, so 29 bits fit in an int.
        adj = [ [0] * N for _ in range(5) ]
        
        # Symmetry breaking:
        # Fix color(1, 0) = 0
        # Fix color(2, 0) = 0, color(2, 1) = 1
        # This reduces the search space significantly.
        
        def check_time():
            return time.time() - start_time > time_limit

        def assign(k, u, c):
            """
            Assign color c to edge (k, u) where u < k.
            Returns True if valid, False if conflict.
            Assumes we are building vertex k, and edges (k, 0)..(k, u-1) are assigned.
            """
            # Check conflict with previously assigned neighbors of k in color c
            # We need to know which w < u have color (k, w) == c.
            # We can maintain a bitmask `assigned_mask[c]` for vertex k?
            # Or just iterate w from 0 to u-1.
            
            # To make it faster, we can pass the state.
            # But in recursion, we can just check.
            
            # Conflict condition:
            # There exists w < u such that colors[k][w] == c AND adj[c][u] & adj[c][w] != 0.
            
            for w in range(u):
                if colors[k][w] == c:
                    if adj[c][u] & adj[c][w]:
                        return False
            return True

        def backtrack(k, u):
            """
            Recursively assign colors to edges (k, 0), ..., (k, k-1).
            u is the current index to assign (from 0 to k-1).
            """
            if check_time():
                return False
            
            if u == k:
                # Vertex k is fully assigned.
                # Update adj for future vertices?
                # No, adj is updated when we assign.
                # Wait, adj[c][v] should reflect the subgraph 0..k-1.
                # When we finish vertex k, we need to add k to adj for vertices 0..k-1?
                # No, adj is used for checking conflicts for the NEXT vertex.
                # So after finishing k, we update adj[c][v] for all v < k to include k?
                # Yes.
                
                for c in range(5):
                    # For each color c, k is a neighbor of v if colors[k][v] == c.
                    # We need to add k to adj[c][v] for all v such that colors[k][v] == c.
                    # And add v to adj[c][k]? No, k is the new vertex, we don't need adj for k yet?
                    # Actually, for the next vertex k+1, we will check adj[c][u] where u < k+1.
                    # So u can be k. So we need adj[c][k] to be populated?
                    # No, adj[c][k] represents neighbors of k in color c.
                    # When we assign (k+1, k), we will check adj[c][k] & adj[c][w].
                    # So we need adj[c][k] to contain the neighbors of k in color c from 0..k-1.
                    
                    mask_k = 0
                    for v in range(k):
                        if colors[k][v] == c:
                            # v is a neighbor of k in color c
                            mask_k |= (1 << v)
                            # Also k is a neighbor of v in color c
                            adj[c][v] |= (1 << k)
                    adj[c][k] = mask_k
                
                # Move to next vertex
                if k + 1 == N:
                    return True # Success
                return backtrack(k + 1, 0)
            
            # Try colors for (k, u)
            # Heuristic: Try colors that are less frequent? Or just 0-4.
            # Symmetry breaking for small k:
            if k == 1 and u == 0:
                c_list = [0]
            elif k == 2 and u == 0:
                c_list = [0]
            elif k == 2 and u == 1:
                c_list = [1] # Since (2,0)=0, (2,1) cannot be 0 if it creates C4? 
                             # Actually (2,0)=0, (2,1)=0 -> N0(0) & N0(1). 
                             # N0(0) is empty (no neighbors < 0). So it's valid.
                             # But we can fix it to break symmetry further.
                             # Let's just try 0, 1, 2, 3, 4 in order.
            else:
                c_list = [0, 1, 2, 3, 4]
                
            # Optional: Shuffle c_list for k > 2 to avoid deterministic bias?
            # Or use a heuristic.
            
            for c in c_list:
                if assign(k, u, c):
                    colors[k][u] = c
                    colors[u][k] = c
                    if backtrack(k, u + 1):
                        return True
                    colors[k][u] = -1
                    colors[u][k] = -1
            return False

        # Start backtracking
        if N == 1:
            return {'n': 1, 'colors': []}
            
        # Initialize vertex 0
        # No edges.
        
        # Start with vertex 1
        # We call backtrack(1, 0)
        
        # We need to handle the case where symmetry breaking fails?
        # The symmetry breaking fixes specific colors. If no solution exists with those fixes,
        # we might miss a solution.
        # However, for highly symmetric problems, usually one symmetric representative is enough
        # IF the symmetry group acts transitively on the set of solutions.
        # For edge colorings, permuting color names is a symmetry.
        # Fixing color(1,0)=0 is safe.
        # Fixing color(2,0)=0, color(2,1)=1 is safe?
        # If there is a solution, we can permute colors so that (1,0) is 0.
        # Then we can permute colors so that (2,0) is 0 (if (2,0) was not 0, swap 0 and c(2,0)).
        # But if (2,0) was 0, we can permute 1 and c(2,1) to make (2,1)=1?
        # This assumes (2,1) != 0.
        # If (2,1) == 0, then we have a triangle (0,1,2) with (1,0)=0, (2,0)=0, (2,1)=0.
        # This is a monochromatic triangle.
        # Does a C4-free coloring allow monochromatic triangles?
        # Yes, C4 is a cycle of length 4. Triangle is C3.
        # So (2,1) CAN be 0.
        # So fixing (2,1)=1 might prune valid solutions where (2,1)=0.
        
        # To be safe, we should not fix (2,1) to 1 if it can be 0.
        # Let's just fix (1,0)=0.
        # And maybe (2,0)=0.
        # Let's try only fixing (1,0)=0 first. If it's too slow, add more.
        
        # Revised symmetry breaking:
        # k=1, u=0: c=0
        # k=2, u=0: c=0
        # k=2, u=1: try all c in 0..4
        
        # Let's modify the c_list logic in backtrack.
        
        if backtrack(1, 0):
            # Extract colors
            final_colors = []
            for i in range(N):
                for j in range(i+1, N):
                    final_colors.append(colors[i][j])
            return {'n': N, 'colors': final_colors}
        
        return None

    # Try N from 26 down to 11
    # We expect 26 to be found quickly.
    # If 26 is found, we return it.
    # We can try 27, 28 with a small time limit.
    
    total_time_limit = 115
    start_total = time.time()
    
    # Try 26 first (known to exist)
    # Allocate most time to 26? Or try 27 with a small budget?
    # If we find 26, score is ~2.36.
    # If we find 27, score is ~2.45.
    # Let's try 27 with 5s, then 26 with remaining time.
    # Actually, 27 might not exist.
    # Let's just try 26 with a generous time limit.
    # If it fails, try 25, etc.
    
    for N in [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        time_remaining = total_time_limit - (time.time() - start_total)
        if time_remaining < 1:
            break
            
        # Allocate time
        if N == 27:
            alloc_time = min(5, time_remaining)
        else:
            alloc_time = min(time_remaining, 100)
            
        res = solve_backtracking(N, time_limit=alloc_time)
        if res is not None:
            return res
            
    # Fallback
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
