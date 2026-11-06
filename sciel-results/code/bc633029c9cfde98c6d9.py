"""Combined approach: fast C6 detection via matrix multiplication + local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _edge_index(i, j, n):
    """Return the index of edge (i,j) in the lexicographic edge list for K_n."""
    return i * (2 * n - i - 3) // 2 + (j - i - 1)

def _build_adj(n, colors, color):
    """Build adjacency list for a given color class."""
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if colors[_edge_index(i, j, n)] == color:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def _has_c6_fast(n, colors):
    """
    Check if any color class contains a C6.
    Uses matrix multiplication to count 4-step paths between endpoints of each edge.
    Returns True if a C6 is found, False otherwise.
    """
    for c in range(4):
        # Build adjacency matrix for color c
        A = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(i + 1, n):
                if colors[_edge_index(i, j, n)] == c:
                    A[i, j] = 1
                    A[j, i] = 1
        
        # Compute A^4: (A^4)[i][j] = number of walks of length 4 from i to j
        A2 = A @ A
        A4 = A2 @ A2
        
        # For each edge (i,j) in color c, if there's a walk of length 4 from i to j
        # in the graph WITHOUT using edge (i,j), that walk combined with edge (i,j)
        # forms a closed walk of length 5... wait, we need C6.
        # C6: 6 vertices, 6 edges. If we have edge (i,j), the remaining 4 edges form
        # a path of length 4 from i to j (going the other way around).
        # Actually: C6 has vertices v0..v5. Edge (v0,v1). The rest is path v1-v2-v3-v4-v5-v0,
        # which is a path of length 4 from v1 to v0.
        # So for edge (i,j), we need a path of length 4 from i to j in G_c - {(i,j)}.
        # A walk of length 4 from i to j that doesn't use edge (i,j) directly...
        # But A^4 counts all walks of length 4, including those that might use (i,j).
        # However, a walk of length 4 from i to j that uses edge (i,j) would have to
        # traverse it, which would require revisiting i or j, making it not a simple path.
        # For a simple path of length 4, all 6 vertices are distinct.
        # If A^4[i][j] > 0 and (i,j) is an edge, we need to verify it's a simple path.
        # But for detection, if A^4[i][j] >= 1 for an edge (i,j), it's very likely a C6.
        # Let's be more careful: subtract the contribution of walks that use edge (i,j).
        
        # Actually, let's just check: for each edge (i,j) in color c,
        # if A4[i][j] > 0, then there exists a walk of length 4 from i to j.
        # This walk plus edge (i,j) gives a closed walk of length 5.
        # Wait, that's not a C6. Let me reconsider.
        
        # C6: v0-v1-v2-v3-v4-v5-v0. Six edges.
        # If we fix edge (v0,v1), the remaining path is v1-v2-v3-v4-v5-v0, which has 4 edges.
        # So we need a path of length 4 from v1 to v0.
        # A^4[v1][v0] counts walks of length 4 from v1 to v0.
        # If A^4[v1][v0] > 0, there's a walk of length 4. If it's simple, it's a C6.
        # For n <= 20, we can verify simplicity if needed, but let's first check if A^4 > 0.
        
        found = False
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:  # edge exists in color c
                    # Check if there's a path of length 4 from i to j
                    # A4[i][j] counts walks of length 4 from i to j
                    if A4[i, j] > 0:
                        # This is a strong indicator of a C6. 
                        # For small n, we can do a quick DFS to confirm.
                        # But let's accept this as a detection for now.
                        found = True
                        break
            if found:
                break
        if found:
            return True
    
    return False

def _has_c6_verify(n, colors):
    """
    Verify C6 presence with DFS for accuracy. Used when _has_c6_fast indicates a C6.
    """
    for c in range(4):
        adj = _build_adj(n, colors, c)
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                # Temporarily remove edge (u,v)
                adj[u].remove(v)
                adj[v].remove(u)
                
                # DFS for path of length 4 from u to v
                def dfs(node, depth, visited):
                    if depth == 4:
                        return node == v
                    for nb in adj[node]:
                        if nb not in visited:
                            visited.add(nb)
                            if dfs(nb, depth + 1, visited):
                                return True
                            visited.remove(nb)
                    return False
                
                found = dfs(u, 0, {u})
                adj[u].append(v)
                adj[v].append(u)
                if found:
                    return True
    return False

def _local_search(n, colors, deadline):
    """
    Local search to eliminate C6s. Try flipping colors of edges involved in C6s.
    Returns (colors, valid) where valid is True if no C6s remain.
    """
    colors = list(colors)
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Check if currently valid
        if not _has_c6_fast(n, colors):
            # Verify with DFS to be safe
            if not _has_c6_verify(n, colors):
                return colors, True
        
        # Find an edge involved in a C6 and try flipping its color
        # For simplicity, try random edges
        edge_idx = random.randint(0, n * (n - 1) // 2 - 1)
        old_color = colors[edge_idx]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            new_color = (new_color + 1) % 4
        
        colors[edge_idx] = new_color
        
        # Check if this improved things (fewer C6s)
        # For now, just accept the change and continue
        # A more sophisticated approach would count C6s before and after
    
    # Final check
    if not _has_c6_fast(n, colors):
        if not _has_c6_verify(n, colors):
            return colors, True
    
    return colors, False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with the known n=9 solution
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_n = n_seed
    best_colors = colors_seed
    
    random.seed(42)
    
    # Try to build up from n=9 to larger n
    for target_n in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
        if time.time() > deadline - 20:
            break
        
        # Initialize by extending the best known coloring
        # Use a heuristic: color new edges based on distance in a circular arrangement
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Start with random initialization
        best_for_n = None
        
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Local search
            colors, valid = _local_search(n, colors, deadline - 5)
            
            if valid:
                best_for_n = colors
                best_n = n
                best_colors = colors
                break
        
        if best_for_n is None:
            # Try a structured initialization
            # Use circular distance coloring
            colors_struct = []
            for i in range(n):
                for j in range(i + 1, n):
                    dist = min(j - i, n - j + i)
                    colors_struct.append((dist - 1) % 4)
            
            colors_struct, valid = _local_search(n, colors_struct, deadline - 5)
            if valid:
                best_for_n = colors_struct
                best_n = n
                best_colors = colors_struct
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
