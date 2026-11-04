<<<<<<< SEARCH
def _girth_at_least(n, edges, g):
    """
    Check if the girth of the graph is at least g.
    Returns True if no cycle of length < g exists.
    Uses non-backtracking BFS from each vertex to certify girth.
    """
    if g <= 1:
        return True
        
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each start vertex, do non-backtracking BFS
    # Track (vertex, previous_vertex) states
    # If we reach a state where vertex == start and depth >= 2, we found a cycle
    # Actually, we need to detect any cycle of length < g
    
    for start in range(n):
        # State: (current_vertex, prev_vertex)
        # dist[(v, p)] = length of non-backtracking walk from (start, None) to (v, p)
        # We want to find if there's a cycle of length < g
        
        # BFS from (start, -1) where -1 means no previous
        # dist[(v, p)] = shortest non-backtracking path length from start to v coming from p
        dist = {}
        queue = [(start, -1, 0)]  # (vertex, prev, depth)
        dist[(start, -1)] = 0
        head = 0
        
        while head < len(queue):
            u, prev_u, depth = queue[head]
            head += 1
            
            if depth >= g - 1:
                continue
            
            for v in adj[u]:
                if v == prev_u:
                    continue  # Non-backtracking
                
                new_depth = depth + 1
                state = (v, u)
                
                if state in dist:
                    # We've seen this state before
                    # Check if this creates a short cycle
                    # The cycle would be: start -> ... -> u -> v -> ... -> (v, u) already visited
                    # Actually, if we revisit (v, u), it means there's a cycle
                    # The cycle length is at least 2 (but we skip backtracking, so min is 3)
                    # We need to be careful here
                    
                    # If v == start and new_depth >= 2, we found a cycle of length new_depth
                    if v == start and new_depth >= 2:
                        if new_depth < g:
                            return False
                else:
                    dist[state] = new_depth
                    queue.append((v, u, new_depth))
                    
                    # If we reach start with depth >= 2, we found a cycle
                    if v == start and new_depth >= 2:
                        if new_depth < g:
                            return False
    
    return True
=======
def _girth_at_least_fast(n, edges, g):
    """
    Check if the girth of the graph is at least g using vectorized non-backtracking BFS.
    Returns True if no cycle of length < g exists.
    """
    if g <= 1:
        return True
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each start vertex, do non-backtracking BFS
    # We track states (v, prev) where prev is the vertex we came from
    # A cycle of length L < g exists iff we can reach (start, some_prev) at depth L-1
    # Actually, we detect cycles by checking if we revisit (v, u) where u is in the path
    
    # More efficient: for each start, BFS non-backtracking, and check if we return to start
    # with depth < g
    
    for start in range(n):
        # dist[v][prev] = shortest non-backtracking path length from start to v coming from prev
        # Use a 2D array or dict
        # n can be up to ~1250, degree is k=5, so states are at most n*k = 6250
        # Use a flat array indexed by v * k + prev_index_in_adj[v]
        # But prev varies, so let's use a dict or 2D array
        
        # Better: use a 2D numpy array dist[n][n] but that's n^2 = 1.5M for n=1250, which is fine
        # Actually, we only need dist for (v, prev) where prev is an actual neighbor
        # Let's use a dictionary with (v, prev) keys, but vectorize the BFS
        
        # Alternative: use BFS level by level with arrays
        # State: (vertex, prev_vertex)
        # We'll use a 2D boolean array visited[v][prev] and track depth
        
        # For efficiency, use a list of (v, prev, depth) but process level by level
        # Actually, let's just do it with a simple BFS but using lists instead of dicts
        
        visited = [[False] * n for _ in range(n)]  # visited[v][prev]
        # This is n^2 = 1.5M booleans for n=1250, which is ~1.5MB, acceptable
        
        queue = [(start, -1, 0)]
        visited[start][-1] = True  # Mark start with prev=-1 as visited
        head = 0
        
        while head < len(queue):
            u, prev_u, depth = queue[head]
            head += 1
            
            if depth >= g - 1:
                continue
            
            for v in adj[u]:
                if v == prev_u:
                    continue
                
                if visited[v][u]:
                    # We've seen this state before
                    # Check if this creates a short cycle
                    if v == start and depth + 1 < g:
                        return False
                else:
                    visited[v][u] = True
                    queue.append((v, u, depth + 1))
                    
                    if v == start and depth + 1 < g:
                        return False
    
    return True
>>>>>>> REPLACE
