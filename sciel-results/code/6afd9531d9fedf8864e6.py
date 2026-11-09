import time
import numpy as np
import sys

def _check_k222_in_subgraph(adj, S, c):
    """
    Check if the subgraph induced by vertices in S contains a K2,2 in color c.
    adj is a 2D numpy array of colors.
    S is a list of vertices.
    Returns True if K2,2 found.
    """
    n = len(S)
    if n < 4:
        return False
    
    # Map original indices to 0..n-1 for local adjacency
    # But we can just use the global adj array.
    
    # For each pair (u, v) in S, check common neighbors in S
    # If any pair has >= 2 common neighbors in S, then K2,2 exists.
    # Wait, K2,2 is {u, v} and {w, x} such that edges (u,w), (u,x), (v,w), (v,x) exist.
    # This is equivalent to: there exist u, v in S such that |N_c(u) \cap N_c(v) \cap S| >= 2.
    
    for i in range(n):
        u = S[i]
        for j in range(i + 1, n):
            v = S[j]
            # Count common neighbors in S
            count = 0
            for k in range(n):
                w = S[k]
                if w == u or w == v:
                    continue
                if adj[u][w] == c and adj[v][w] == c:
                    count += 1
                    if count >= 2:
                        return True
    return False

def _is_valid_extension(adj, u, v, c, n):
    """
    Check if adding edge (u, v) with color c to the current partial graph
    (vertices 0..n-1) creates a monochromatic K2,2,2 of color c.
    """
    # The new K2,2,2 must involve the new edge (u, v).
    # The other two parts must be pairs of vertices from {0, ..., n-1} \ {u, v}.
    # Let S be the set of vertices w such that edge (u, w) is color c AND edge (v, w) is color c.
    # If such a K2,2,2 exists, it consists of {u, v} and two disjoint pairs {a, b} and {d, e} from S,
    # such that all edges between {a, b} and {d, e} are color c.
    # This is equivalent to checking if the subgraph of color c induced by S contains a K2,2.
    
    S = []
    for w in range(n):
        if w == u or w == v:
            continue
        if adj[u][w] == c and adj[v][w] == c:
            S.append(w)
            
    if _check_k222_in_subgraph(adj, S, c):
        return False
    return True

def _backtrack_search(n, time_limit):
    """
    Try to find a valid coloring for K_n using backtracking.
    """
    start_time = time.time()
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Symmetry breaking: fix first few edges to 0
    # e(0,1)=0, e(0,2)=0, e(0,3)=0
    # This breaks the color symmetry and some vertex symmetries.
    fixed_edges = [(0, 1, 0), (0, 2, 0), (0, 3, 0)]
    for u, v, c in fixed_edges:
        adj[u][v] = c
        adj[v][u] = c
        
    # We will assign edges in lexicographic order (i, j) with i < j.
    # We need to skip the fixed edges.
    
    # Precompute list of edges to assign
    edges_to_assign = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in [(0, 1), (0, 2), (0, 3)]:
                continue
            edges_to_assign.append((i, j))
            
    # Recursive search
    def dfs(idx):
        if time.time() - start_time > time_limit:
            return False
            
        if idx == len(edges_to_assign):
            return True
            
        u, v = edges_to_assign[idx]
        
        # Try color 0
        if _is_valid_extension(adj, u, v, 0, n):
            adj[u][v] = 0
            adj[v][u] = 0
            if dfs(idx + 1):
                return True
            adj[u][v] = 0 # reset? No, it's 0.
            adj[v][u] = 0
            
        # Try color 1
        if _is_valid_extension(adj, u, v, 1, n):
            adj[u][v] = 1
            adj[v][u] = 1
            if dfs(idx + 1):
                return True
            adj[u][v] = 0 # reset to 0? No, we need to unset it.
            adj[v][u] = 0
            
        return False

    # Increase recursion limit if necessary, but n=30 is small enough for edges ~ 435.
    # However, recursion depth is 435, which is fine.
    
    success = dfs(0)
    
    if success:
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(adj[i][j]))
        return n, colors
    return None

def _paley_coloring(p):
    """Generate Paley coloring for prime p ≡ 1 (mod 4)."""
    squares = set(x * x % p for x in range(1, p))
    adj = np.zeros((p, p), dtype=np.int8)
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            c = 0 if (j - i) % p in squares else 1
            colors.append(c)
            adj[i][j] = c
            adj[j][i] = c
    return p, colors, adj

def construct():
    start_time = time.time()
    # Allow 100 seconds for search, leaving 20s for safety and fallback
    time_limit = 100.0
    
    # Try to find a 30-vertex coloring
    result = _backtrack_search(30, time_limit)
    
    if result is not None:
        n, colors = result
        return {'n': n, 'colors': colors}
        
    # Fallback to Paley 29
    n, colors, adj = _paley_coloring(29)
    return {'n': n, 'colors': colors}
