import random
import time
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_edges = []
    
    # Helper to check if adding (u, v) creates a C6
    # This is equivalent to checking if there is a path of length 5 between u and v
    def has_path_len_5(u, v, adj):
        # BFS from u, limit depth to 5
        # We want to know if v is reachable in exactly 5 steps.
        # Actually, if v is reachable in 5 steps, adding the edge creates a C6.
        # If v is reachable in < 5 steps, adding the edge creates a shorter cycle, which is allowed.
        # So we specifically need to check for existence of a path of length 5.
        
        # Optimization: If dist(u, v) < 5, we still need to check if there is a path of length 5.
        # A path of length 5 implies dist(u, v) <= 5.
        # We can just do BFS and record distances. If dist[u][v] <= 5, it's not enough.
        # We need to know if there is a walk of length 5? No, a simple path.
        # A cycle must be simple. So we need a simple path of length 5.
        
        # BFS gives shortest path. If shortest path < 5, there might still be a longer simple path of length 5.
        # Example: u-a-v (len 2). Also u-b-c-d-e-v (len 5).
        # So BFS distance is not sufficient. We need to check existence of a simple path of length 5.
        
        # Since N=40, we can use a recursive DFS or iterative deepening?
        # Or just standard DFS with visited set and depth limit.
        
        visited = [False] * n
        visited[u] = True
        
        def dfs(curr, depth):
            if depth == 5:
                return curr == v
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    if dfs(neighbor, depth + 1):
                        return True
                    visited[neighbor] = False
            return False
            
        return dfs(u, 0)

    # This DFS might be slow if called 780 times * many restarts.
    # Let's optimize the check.
    # Precomputing all pairs shortest paths doesn't help with "path of length exactly 5".
    # However, for N=40, the DFS is bounded by $N^5$ in worst case, but practically much smaller due to pruning.
    # Average degree will be around 5-10.
    # Branching factor ~5. Depth 5. $5^5 = 3125$ nodes visited per check.
    # 780 edges * 3125 ops = ~2.4 million ops per restart.
    # In Python, this might take 1-2 seconds per restart.
    # We have 110 seconds. So ~50-100 restarts.
    # This should be sufficient to find a good solution.

    def greedy_randomized():
        adj = [[False] * n for _ in range(n)]
        edges = []
        
        # List of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a C6
            # i.e., is there a simple path of length 5 between u and v in current graph?
            
            # Quick check: if u and v are not connected in 5 steps, safe.
            # We use the DFS above.
            
            # To speed up, we can check if u and v are in the same component and dist < 5?
            # No, as discussed, dist < 5 doesn't preclude a length 5 path.
            
            # Let's use the DFS.
            # Optimization: Pass adj as list of lists of neighbors for faster iteration
            # Current adj is matrix. Let's maintain neighbor lists.
            
            # Actually, let's rewrite the check to use neighbor lists for speed.
            pass

    # Let's refine the inner loop for speed.
    def construct_greedy():
        adj = [[] for _ in range(n)]  # List of neighbors
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if path of length 5 exists between u and v
            # Using iterative DFS to avoid recursion overhead/limit
            stack = [(u, 0, [u])]  # (node, depth, path)
            # This path list is expensive. 
            # Better: visited array with timestamps or sets.
            # Since depth is only 5, we can use a set of visited nodes for the current DFS branch?
            # Or just a global visited array for the DFS run.
            
            # Let's use a simple recursive function with a visited set.
            # Python recursion limit is usually 1000, depth 5 is fine.
            
            visited = set()
            visited.add(u)
            
            def dfs(curr, depth):
                if depth == 5:
                    return curr == v
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if dfs(neighbor, depth + 1):
                            return True
                        visited.remove(neighbor)
                return False

            if not dfs(u, 0):
                # Safe to add
                adj[u].append(v)
                adj[v].append(u)
                edges.append([u, v])
                
        return edges

    best_count = 0
    
    while time.time() < deadline:
        edges = construct_greedy()
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = edges.copy()
            
    return best_edges

# The above might be too slow in pure Python if the DFS is called too often.
# Let's optimize the DFS check.
# Instead of checking every edge with a full DFS, we can be smarter.
# But given N=40, it's likely fine.
# Let's add a quick check: if u and v are not connected at all, or connected by path < 5?
# No, we need to rule out path of length 5.
# If dist(u, v) > 5, then no path of length 5 exists? 
# No, shortest path > 5 implies no path of length 5.
# So if dist(u, v) > 5, it's SAFE to add.
# We can compute dist(u, v) quickly with BFS.
# If dist(u, v) > 5, add immediately.
# If dist(u, v) <= 5, we need to check if there is a path of length 5.
# This is a "longer path" problem.
# For small N, we can just do the DFS.

# Revised construct_greedy with dist check
def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_edges = []
    best_count = 0

    def get_dist(u, v, adj):
        # BFS to find shortest distance
        if u == v: return 0
        visited = [-1] * n
        visited[u] = 0
        queue = [u]
        while queue:
            curr = queue.pop(0)
            if curr == v:
                return visited[curr]
            for neighbor in adj[curr]:
                if visited[neighbor] == -1:
                    visited[neighbor] = visited[curr] + 1
                    queue.append(neighbor)
        return -1

    def has_path_len_5(u, v, adj):
        # Only called if dist(u, v) <= 5
        # Check for simple path of length 5
        visited = set()
        visited.add(u)
        
        def dfs(curr, depth):
            if depth == 5:
                return curr == v
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if dfs(neighbor, depth + 1):
                        return True
                    visited.remove(neighbor)
            return False
        return dfs(u, 0)

    def construct_greedy():
        adj = [[] for _ in range(n)]
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Quick dist check
            d = get_dist(u, v, adj)
            if d == -1:
                # Not connected, safe
                adj[u].append(v)
                adj[v].append(u)
                edges.append([u, v])
            elif d > 5:
                # Shortest path > 5, so no path of length 5, safe
                adj[u].append(v)
                adj[v].append(u)
                edges.append([u, v])
            else:
                # d <= 5, need to check for path of length 5
                if not has_path_len_5(u, v, adj):
                    adj[u].append(v)
                    adj[v].append(u)
                    edges.append([u, v])
                    
        return edges

    while time.time() < deadline:
        edges = construct_greedy()
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = edges.copy()
            
    return best_edges
