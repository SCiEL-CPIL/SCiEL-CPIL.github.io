import time
import sys
from collections import deque

# Increase recursion limit for DFS
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    best_path = []
    
    # Helper to check time
    def time_check():
        return time.time() > deadline

    # Strategy 1: Greedy Warnsdorff to get a baseline
    for start in range(n):
        if time_check():
            break
        path = [start]
        visited = {start}
        while True:
            curr = path[-1]
            candidates = [v for v in adj[curr] if v not in visited]
            if not candidates:
                break
            # Sort by number of unvisited neighbors (Warnsdorff)
            candidates.sort(key=lambda v: len(adj[v] - visited))
            next_v = candidates[0]
            path.append(next_v)
            visited.add(next_v)
        
        if len(path) > len(best_path):
            best_path = path[:]
    
    if len(best_path) == n:
        return best_path

    # Strategy 1.5: Bipartite matching relaxation to identify bottlenecks
    # This helps identify vertices that are "forced" into certain positions
    # by the structure of the graph, which can guide the DFS ordering.
    
    # Compute a maximum matching in the graph using Hopcroft-Karp style
    # We treat the graph as bipartite by splitting each vertex into left/right copies
    # and matching across edges. This gives us a lower bound on the number of
    # internal path vertices we can accommodate.
    
    def compute_matching_slack(adj, n):
        """Compute matching-based slack for each vertex.
        
        Returns a list where slack[v] indicates how 'constrained' vertex v is.
        Lower slack means more constrained (should be prioritized in DFS).
        """
        # For a path, internal vertices need 2 incident path edges,
        # endpoints need 1. We use matching to estimate feasibility.
        
        # Build a bipartite representation: for each edge (u,v), we can match
        # u's "left" copy to v's "right" copy or vice versa.
        # A simple approach: compute max matching in the original graph.
        
        # Use a simple augmenting path algorithm for bipartite matching
        # on a bipartite double cover of the graph.
        
        # Actually, let's use a simpler heuristic: compute the degree-based
        # constraint. Vertices with low degree relative to their component size
        # are more constrained.
        
        # Find connected components
        visited_comp = [False] * n
        components = []
        for i in range(n):
            if not visited_comp[i]:
                comp = []
                queue = deque([i])
                visited_comp[i] = True
                while queue:
                    u = queue.popleft()
                    comp.append(u)
                    for v in adj[u]:
                        if not visited_comp[v]:
                            visited_comp[v] = True
                            queue.append(v)
                components.append(comp)
        
        # For each component, compute a slack score
        # A path through k vertices needs k-1 edges.
        # If the component has c vertices and e edges, and we want a Hamiltonian path,
        # we need at least c-1 edges in the path.
        # Slack = (available edges) - (required edges)
        
        slack = [0.0] * n
        for comp in components:
            c = len(comp)
            if c <= 1:
                continue
            # Count edges within the component
            e_count = 0
            for u in comp:
                for v in adj[u]:
                    if v in comp and u < v:
                        e_count += 1
            
            # Required edges for a Hamiltonian path
            required = c - 1
            # Available edges (we can use at most c-1 in the path)
            available = min(e_count, required)
            
            # Component-level slack
            comp_slack = available - required  # This will be <= 0 if not enough edges
            
            # For each vertex in the component, compute individual constraint
            # based on its degree within the component
            for u in comp:
                deg_in_comp = len([v for v in adj[u] if v in comp])
                # A vertex with low degree is more constrained
                # Normalize: vertices with degree 1 are most constrained
                if deg_in_comp == 0:
                    slack[u] = 1.0  # Isolated, very constrained
                elif deg_in_comp == 1:
                    slack[u] = 0.3  # Endpoint candidate
                elif deg_in_comp == 2:
                    slack[u] = 0.6  # Could be internal
                else:
                    slack[u] = 0.9  # Flexible
                # Adjust by component slack
                slack[u] += comp_slack * 0.1
        
        return slack, components

    slack, components = compute_matching_slack(adj, n)
    
    # Strategy 1.6: Try to extend the best path by inserting unvisited vertices
    def try_extend_path(path, adj, n, slack, deadline):
        """Try to extend a path by inserting unvisited vertices."""
        if not path:
            return path
        
        visited = set(path)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Try to insert an unvisited vertex w between u and v
                # such that u-w and w-v are both edges
                candidates = [w for w in adj[u] if w not in visited and w in adj[v]]
                if candidates:
                    # Choose the candidate with lowest slack (most constrained)
                    candidates.sort(key=lambda w: slack[w])
                    w = candidates[0]
                    path.insert(i+1, w)
                    visited.add(w)
                    improved = True
                    break
            # Also try to extend endpoints
            # Extend start
            start = path[0]
            ext_candidates = [w for w in adj[start] if w not in visited]
            if ext_candidates:
                ext_candidates.sort(key=lambda w: slack[w])
                path.insert(0, ext_candidates[0])
                visited.add(ext_candidates[0])
                improved = True
            # Extend end
            end = path[-1]
            ext_candidates = [w for w in adj[end] if w not in visited]
            if ext_candidates:
                ext_candidates.sort(key=lambda w: slack[w])
                path.append(ext_candidates[0])
                visited.add(ext_candidates[0])
                improved = True
        
        return path

    # Apply extension to the best path found so far
    extended_path = try_extend_path(best_path[:], adj, n, slack, deadline)
    if len(extended_path) > len(best_path):
        best_path = extended_path
    
    if len(best_path) == n:
        return best_path

    # Strategy 2: DFS with Warnsdorff heuristic
    # We try to find a path longer than best_path
    # We use recursion with backtracking
    
    visited = [False] * n
    path = []
    
    # Precompute slack for ordering (already computed above)
    # Use slack to guide DFS: prioritize vertices with lower slack (more constrained)
    
    def dfs(u, depth=0):
        nonlocal best_path, time_check, deadline
        if time_check():
            return False
        
        path.append(u)
        visited[u] = True
        
        if len(path) > len(best_path):
            best_path = path[:]
            
        if len(path) == n:
            return True
        
        curr = u
        unvisited_set = {x for x in range(n) if not visited[x]}
        candidates = [v for v in adj[curr] if not visited[v]]
        
        if not candidates:
            path.pop()
            visited[u] = False
            return False
        
        # Enhanced ordering: combine Warnsdorff (fewer unvisited neighbors) with slack
        # Vertices with lower slack are more constrained and should be visited earlier
        def order_key(v):
            remaining_deg = len(adj[v] & unvisited_set)
            # Lower remaining degree is better (Warnsdorff)
            # Lower slack is better (more constrained)
            # Combine: primarily by remaining degree, secondarily by slack
            return (remaining_deg, slack[v])
        
        candidates.sort(key=order_key)
        
        for v in candidates:
            if time_check():
                break
            result = dfs(v, depth + 1)
            if result:
                # Found a complete path; unwind without popping
                return True
        
        # If we didn't return True from the loop, we failed to extend
        path.pop()
        visited[u] = False
        return False

    # Start DFS from vertices ordered by slack (most constrained first)
    # This is based on the matching relaxation insight: constrained vertices
    # should be placed early to avoid trapping them
    
    # Get vertices sorted by slack (ascending - most constrained first)
    start_order = sorted(range(n), key=lambda v: slack[v])
    
    # Try DFS from the most constrained vertices first
    for start in start_order:
        if time_check():
            break
        if len(best_path) == n:
            break
        
        # Reset state for new DFS
        visited = [False] * n
        path = []
        
        dfs(start)
        
        if len(best_path) == n:
            return best_path
    
    # If we still haven't found a full path, try a few more starts
    # that weren't tried (shouldn't happen since we tried all, but just in case)
    for start in range(n):
        if time_check():
            break
        if len(best_path) == n:
            break
        if visited[start]:  # Already tried in the slack-ordered pass
            continue
        visited = [False] * n
        path = []
        dfs(start)
    
    return best_path if best_path else [0]
