"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def has_path_length_4(adj, u, v, n):
    """
    Check if there is a path of length exactly 4 between u and v in the graph defined by adj.
    adj is a list of integers (bitmasks) of size n.
    """
    # We want to know if there exist a, b, c such that u-a-b-c-v is a path.
    # This is equivalent to:
    # (adj[u] & adj[adj[v]] & adj[adj[v]] & adj[adj[v]]) ... no, that's not right.
    
    # Path u -> a -> b -> c -> v
    # a in N(u)
    # b in N(a)
    # c in N(b)
    # v in N(c)
    
    # Let's compute the set of vertices reachable from u in exactly 4 steps.
    # Or simpler:
    # Step 1: N(u)
    # Step 2: N(N(u))
    # Step 3: N(N(N(u)))
    # Step 4: N(N(N(N(u))))
    # Check if v is in Step 4.
    
    # However, we must ensure the path is simple? 
    # A cycle of length 6 requires a simple cycle.
    # If there is a walk of length 4, does it imply a simple path?
    # If u and v are distinct, and we have a walk of length 4, it might contain loops.
    # But if we are adding edge (u,v) to form a C6, the path of length 4 must be internally disjoint from u and v.
    # In a simple graph, if dist(u,v) == 4, there is a simple path of length 4.
    # If dist(u,v) < 4, there might still be a walk of length 4, but no simple path of length 4 that doesn't use shorter paths?
    # Actually, if there is a cycle of length 6 formed by (u,v) and a path of length 4, that path must be simple and not use u or v internally.
    # If the shortest path is < 4, we might have a "shorter" cycle, but we are looking for C6.
    # The condition for (u,v) to complete a C6 is that there is a path of length 4 between u and v in G \ {(u,v)} such that the internal vertices are distinct and not u,v.
    # For n=17, checking if v is in the set of vertices reachable from u in exactly 4 steps is a good approximation, 
    # but we must ensure the path is simple.
    
    # A safer check:
    # Find all paths of length 4 from u to v.
    # Since n is small, we can just do a DFS from u with depth limit 4, avoiding revisiting nodes.
    
    # Optimization: 
    # If u and v are not connected by a path of length 4, no C6 is formed.
    
    def dfs(node, depth, visited):
        if depth == 4:
            return (v == node)
        if depth > 4:
            return False
        neighbors = adj[node]
        while neighbors:
            lsb = neighbors & -neighbors
            nxt = lsb.bit_length() - 1
            neighbors ^= lsb
            if nxt not in visited:
                visited.add(nxt)
                if dfs(nxt, depth + 1, visited):
                    return True
                visited.remove(nxt)
        return False
        
    # Start from u
    # We need to reach v in 4 steps.
    # To speed up, we can prune: 
    # If we are at depth d, we need 4-d more steps to reach v.
    # So we only explore neighbors of current node that can reach v in the remaining steps.
    # This is getting complex. For n=17, the simple DFS above is fast enough.
    
    visited = {u}
    return dfs(u, 0, visited)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # We try to find a valid coloring for n=17
    # If it fails, we try n=16, etc.
    
    def solve(n):
        if time.time() > deadline:
            return None
        
        # adj[c] is a list of n integers (bitmasks)
        adj = [[0] * n for _ in range(4)]
        colors = [0] * (n * (n - 1) // 2)
        
        # List of all edges (i, j) with i < j
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # To improve performance, we can order edges. 
        # However, static ordering might be okay. 
        # Dynamic ordering (most constrained first) is better but harder to implement with backtracking stack.
        # Let's stick to lexicographic order first, but maybe shuffle?
        # Randomized restarts might help.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(edges):
                return True
            
            u, v = edges[idx]
            
            # Check if we can assign any color
            possible_colors = []
            for c in range(4):
                # Check if u and v are already connected in color c (shouldn't be if we are adding it)
                # Check if adding (u,v) with color c creates a C6
                # This happens if there is a path of length 4 between u and v in G_c
                
                # Optimization: If u and v are not connected in the subgraph of color c, 
                # we can check for path of length 4.
                
                # We need to check if there is a simple path of length 4 between u and v in adj[c].
                # We can do a quick check: 
                # If (adj[c][u] & adj[c][v]) == 0, they don't share neighbors.
                # Actually, let's just run the DFS. It's fast for n=17.
                
                if has_path_length_4(adj[c], u, v, n):
                    continue
                else:
                    possible_colors.append(c)
            
            if not possible_colors:
                return False
            
            # Heuristic: Try colors that are least constrained? 
            # For now, randomize to break symmetry
            random.shuffle(possible_colors)
            
            for c in possible_colors:
                # Assign color c to (u, v)
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                colors[idx] = c
                
                if backtrack(idx + 1):
                    return True
                
                # Unassign
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[idx] = 0
            
            return False

        # Try to solve
        # We might need multiple attempts if the first randomization fails
        # But backtracking should find it if it exists.
        
        # To ensure we don't get stuck in a bad branch, we can add a limit on backtracking steps
        # or just rely on the time deadline.
        
        if backtrack(0):
            return n, colors
        else:
            return None

    # Try n=17
    result = solve(17)
    if result:
        n, colors = result
        return {'n': n, 'colors': colors}
        
    # Try n=16
    result = solve(16)
    if result:
        n, colors = result
        return {'n': n, 'colors': colors}
        
    # Try n=15
    result = solve(15)
    if result:
        n, colors = result
        return {'n': n, 'colors': colors}
        
    # Fallback to a known valid n=9 (from seed)
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

# EVOLVE-BLOCK-END
