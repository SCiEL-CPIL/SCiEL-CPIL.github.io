"""Combined approach: greedy construction with forbidden-pair pruning for C6-free 4-coloring."""
import time
import random
from itertools import combinations

def _has_c6_in_color(edges_by_color, n, color):
    """Check if a specific color class contains a C6. Returns True if found."""
    # Build adjacency for this color
    adj = [set() for _ in range(n)]
    for (u, v) in edges_by_color[color]:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check all 6-tuples for a cycle
    # A C6 is v0-v1-v2-v3-v4-v5-v0 with all distinct vertices
    # We can enumerate combinations of 6 vertices and check if they form a C6
    for vertices in combinations(range(n), 6):
        # Check all 6! / 12 = 60 possible C6 orderings on these 6 vertices
        # Actually, just check if there's a Hamiltonian cycle in the induced subgraph
        v0, v1, v2, v3, v4, v5 = vertices
        # Check one specific cycle ordering and its reverse
        if (v1 in adj[v0] and v2 in adj[v1] and v3 in adj[v2] and 
            v4 in adj[v3] and v5 in adj[v4] and v0 in adj[v5]):
            return True
        # Also check the reverse: v0-v5-v4-v3-v2-v1-v0
        # which is the same as above just reversed, so already covered
        # But we need to check other permutations too...
        # This is getting expensive. Let's use a different approach.
        pass
    
    # Better: for each pair of vertices, check if there's a path of length 5
    # Actually, let's just do a BFS/DFS approach for small n
    # For n <= 20, checking all C6 is feasible with optimization
    
    # Enumerate all simple paths of length 5 (6 vertices)
    # Start from each vertex, do DFS of depth 5
    visited_paths = set()
    
    def dfs(start, current, path, depth):
        if depth == 5:
            # Check if start is adjacent to current
            if start in adj[current]:
                # Found a C6
                return True
            return False
        for neighbor in adj[current]:
            if neighbor not in path:
                if dfs(start, neighbor, path + [neighbor], depth + 1):
                    return True
        return False
    
    for start in range(n):
        if dfs(start, start, [start], 0):
            return True
    
    return False

def _is_valid_coloring(colors, n):
    """Check if the coloring has no monochromatic C6."""
    edges_by_color = {c: [] for c in range(4)}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            edges_by_color[c].append((i, j))
            idx += 1
    
    for c in range(4):
        if _has_c6_in_color(edges_by_color, n, c):
            return False
    return True

def _greedy_construct(n, seed=42, time_limit=30):
    """
    Greedy vertex-by-vertex construction.
    For each new vertex k, assign colors to edges (k, 0), ..., (k, k-1)
    such that no C6 is created in any color class.
    """
    start_time = time.time()
    rng = random.Random(seed)
    
    # colors[i][j] = color of edge (i,j), -1 if unassigned
    colors = [[-1] * n for _ in range(n)]
    
    # For each color, maintain set of edges
    edges_by_color = {c: set() for c in range(4)}
    
    def check_c6_creation(color, u, v, current_edges):
        """Check if adding edge (u,v) in 'color' to current_edges creates a C6."""
        # A C6 through (u,v) means there's a path of length 5 from u to v
        # in the existing graph of this color (not using edge (u,v) itself)
        # Build adjacency for this color from current_edges (excluding (u,v))
        adj = [set() for _ in range(n)]
        for (a, b) in current_edges:
            if a == u and b == v:
                continue
            if a == v and b == u:
                continue
            adj[a].add(b)
            adj[b].add(a)
        
        # Check if there's a path of length 5 from u to v
        # BFS/DFS from u, depth 5, see if we reach v
        # Use iterative deepening or just BFS up to depth 5
        
        from collections import deque
        # State: (vertex, depth, path_set)
        # For efficiency, just track visited vertices at each depth
        visited = [set() for _ in range(6)]  # visited[d] = vertices reachable in d steps
        visited[0].add(u)
        
        for depth in range(1, 6):
            for vertex in list(visited[depth - 1]):
                for neighbor in adj[vertex]:
                    if neighbor not in visited[depth]:
                        visited[depth].add(neighbor)
            if depth == 5 and v in visited[5]:
                return True
            if depth == 5:
                break
        
        return v in visited[5]
    
    def try_assign_vertex(k, attempt=0):
        """Try to assign colors to edges (k, 0), ..., (k, k-1)."""
        if time.time() - start_time > time_limit:
            return False
        
        if k == 0:
            return True
        
        if k == 1:
            colors[0][1] = colors[1][0] = 0
            edges_by_color[0].add((0, 1))
            return True
        
        # For vertex k, assign colors to edges (k, 0), ..., (k, k-1)
        # Try different orderings of colors
        color_order = list(range(4))
        if attempt > 0:
            rng.shuffle(color_order)
        
        # Backtrack on the k edges
        assigned = [-1] * k
        
        def backtrack(idx):
            if time.time() - start_time > time_limit:
                return False
            
            if idx == k:
                # All edges assigned, verify no C6 was created
                # (should be guaranteed by the checks above)
                for i in range(k):
                    c = assigned[i]
                    colors[k][i] = colors[i][k] = c
                    edges_by_color[c].add((k, i))
                return True
            
            for c in color_order:
                assigned[idx] = c
                
                # Check if adding edge (k, idx) in color c creates a C6
                # This would happen if there's a path of length 5 from k to idx
                # in the existing color-c graph (using edges already assigned for vertex k
                # and all previous vertices)
                
                # Build the current color-c graph including partially assigned edges for vertex k
                # Edges from previous vertices (already in edges_by_color[c])
                # Plus edges (k, j) for j < idx that are already assigned to color c
                current_edges = set(edges_by_color[c])
                for j in range(idx):
                    if assigned[j] == c:
                        current_edges.add((k, j))
                
                if check_c6_creation(c, k, idx, current_edges):
                    assigned[idx] = -1
                    continue
                
                if backtrack(idx + 1):
                    return True
                
                assigned[idx] = -1
            
            return False
        
        return backtrack(0)
    
    # Build vertex by vertex
    for k in range(n):
        success = False
        for attempt in range(50):
            if try_assign_vertex(k, attempt):
                success = True
                break
            # If failed, we need to backtrack to previous vertices
            # For simplicity, restart the entire construction
            break
        
        if not success:
            # Try a different seed
            return None
    
    # Extract colors in lexicographic order
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            result.append(colors[i][j])
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to build a large coloring
    best_n = 0
    best_colors = None
    
    # Try n from 17 down to 9
    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            
            colors = _greedy_construct(n, seed=seed, time_limit=20)
            if colors is not None:
                # Verify validity
                if _is_valid_coloring(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break
                else:
                    # Invalid, try next seed
                    continue
        
        if found:
            break
    
    if best_n == 0:
        # Fallback to the known n=9 solution
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    
    return {'n': best_n, 'colors': best_colors}
