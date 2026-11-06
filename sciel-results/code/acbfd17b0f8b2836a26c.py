import math
import time

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    N = 256
    nodes = list(range(1, N + 1))
    
    # Build conflict graph
    # conflict[i][j] = True if i and j conflict
    # We only need to check distinct pairs
    # a, b conflict iff (a*b) % (a+b) == 0
    
    # To speed up, we can precompute gcds or just use the property
    # (a*b) % (a+b) == 0
    # This is O(N^2) which is ~65k operations, very fast.
    
    # Let's build adjacency list
    adj = {i: [] for i in nodes}
    for i in range(len(nodes)):
        a = nodes[i]
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            if (a * b) % (a + b) == 0:
                adj[a].append(b)
                adj[b].append(a)
    
    # Check if graph is bipartite
    # Use BFS/DFS coloring
    color = {}
    is_bipartite = True
    for node in nodes:
        if node not in color:
            color[node] = 0
            queue = [node]
            while queue and is_bipartite:
                u = queue.pop(0)
                for v in adj[u]:
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    else:
                        if color[v] == color[u]:
                            is_bipartite = False
                            break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break
    
    if is_bipartite:
        # Use Hopcroft-Karp to find Max Matching
        # Then Max Independent Set = N - Max Matching Size
        
        # Hopcroft-Karp Implementation
        # Left set: nodes with color 0
        # Right set: nodes with color 1
        
        left_nodes = [n for n in nodes if color[n] == 0]
        right_nodes = [n for n in nodes if color[n] == 1]
        
        # Map right nodes to indices for matching array
        right_index = {n: i for i, n in enumerate(right_nodes)}
        
        pair_u = {u: -1 for u in left_nodes}
        pair_v = {v: -1 for v in right_nodes}
        
        def bfs():
            from collections import deque
            dist = {u: 0 for u in left_nodes}
            queue = deque()
            for u in left_nodes:
                if pair_u[u] == -1:
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v in right_index:
                        pu = pair_v[v]
                        if pu == -1:
                            found = True
                        elif dist[pu] == float('inf'):
                            dist[pu] = dist[u] + 1
                            queue.append(pu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                if v in right_index:
                    pu = pair_v[v]
                    if pu == -1 or (dist.get(pu, float('inf')) == dist.get(u, 0) + 1 and dfs(pu)):
                        pair_u[u] = v
                        pair_v[v] = u
                        return True
            dist[u] = float('inf')
            return False
        
        matching_size = 0
        while bfs():
            for u in left_nodes:
                if pair_u[u] == -1:
                    if dfs(u):
                        matching_size += 1
        
        max_independent_size = N - matching_size
        
        # Reconstruct the Max Independent Set
        # Vertices not in the Minimum Vertex Cover are in the Max Independent Set.
        # Min Vertex Cover can be found from the Max Matching.
        # In a bipartite graph, if U and V are the partitions:
        # Let Z be the set of vertices reachable from unmatched vertices in U via alternating paths.
        # Min Vertex Cover = (U \ Z) U (V \cap Z)
        # Max Independent Set = (U \cap Z) U (V \ Z)
        
        # Find Z
        Z = set()
        from collections import deque
        queue = deque()
        for u in left_nodes:
            if pair_u[u] == -1:
                queue.append(u)
                Z.add(u)
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v in right_index:
                    pu = pair_v[v]
                    if pu == -1:
                        # v is unmatched, but we only add matched ones to Z?
                        # Actually, the standard algorithm for finding Z:
                        # Start from unmatched U.
                        # Traverse unmatched edges to V.
                        # Traverse matched edges from V to U.
                        pass # This logic is tricky to implement correctly on the fly without a standard library.
                        
        # Alternative: Just use the property that MIS = N - MCM.
        # We can greedily construct a large independent set and verify.
        # Or, since we have the matching, we can find the MVC.
        
        # Let's just use a greedy heuristic to pick the nodes, as reconstructing MVC is error-prone.
        # Greedy: Sort by degree, pick lowest degree, remove neighbors.
        # This is usually close to optimal.
        
        # Better: Use the fact that we know the size.
        # Let's just return a greedy independent set.
        
        # Actually, if it's bipartite, we can just pick all of U or all of V? No.
        
        # Let's use a simple heuristic:
        # Start with empty set.
        # Iterate nodes, add if no conflict with current set.
        # Then try to improve.
        
        pass # Fall through to heuristic if reconstruction is too complex
    
    # Heuristic: Greedy + Local Search
    # 1. Greedy construction
    # Sort by degree (ascending)
    degrees = {n: len(adj[n]) for n in nodes}
    sorted_nodes = sorted(nodes, key=lambda x: degrees[x])
    
    S = set()
    for n in sorted_nodes:
        conflict = False
        for m in S:
            if m in adj[n]:
                conflict = True
                break
        if not conflict:
            S.add(n)
    
    # 2. Local Search (Hill Climbing)
    # Try to add removed nodes
    removed = [n for n in nodes if n not in S]
    improved = True
    while improved:
        improved = False
        for n in removed:
            if time.time() > deadline:
                break
            conflict = False
            for m in S:
                if m in adj[n]:
                    conflict = True
                    break
            if not conflict:
                S.add(n)
                removed.remove(n)
                improved = True
                break # Restart scan
    
    # 3. Swap Search
    # Try to swap a node in S with a node not in S to increase size?
    # No, we want to maximize size. If we can't add, we can't swap to increase size.
    # Unless we remove multiple to add more?
    # Given the sparsity, the greedy + add-back is likely optimal or very close.
    
    return list(S)

# The above code has a flaw: if it's bipartite, we can solve exactly.
# Let's refine the bipartite solver to be exact.

def construct():
    start_time = time.time()
    deadline = 50.0
    
    N = 256
    nodes = list(range(1, N + 1))
    
    adj = {i: [] for i in nodes}
    for i in range(len(nodes)):
        a = nodes[i]
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            if (a * b) % (a + b) == 0:
                adj[a].append(b)
                adj[b].append(a)
    
    # Check bipartite
    color = {}
    is_bipartite = True
    for node in nodes:
        if node not in color:
            color[node] = 0
            queue = [node]
            while queue and is_bipartite:
                u = queue.pop(0)
                for v in adj[u]:
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    else:
                        if color[v] == color[u]:
                            is_bipartite = False
                            break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break
    
    if is_bipartite:
        # Hopcroft-Karp
        left_nodes = [n for n in nodes if color[n] == 0]
        right_nodes = [n for n in nodes if color[n] == 1]
        right_index = {n: i for i, n in enumerate(right_nodes)}
        
        pair_u = {u: -1 for u in left_nodes}
        pair_v = {v: -1 for v in right_nodes}
        
        from collections import deque
        
        def bfs():
            dist = {u: 0 for u in left_nodes}
            queue = deque()
            for u in left_nodes:
                if pair_u[u] == -1:
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v in right_index:
                        pu = pair_v[v]
                        if pu == -1:
                            found = True
                        elif dist[pu] == float('inf'):
                            dist[pu] = dist[u] + 1
                            queue.append(pu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                if v in right_index:
                    pu = pair_v[v]
                    if pu == -1 or (dist.get(pu, float('inf')) == dist.get(u, 0) + 1 and dfs(pu)):
                        pair_u[u] = v
                        pair_v[v] = u
                        return True
            dist[u] = float('inf')
            return False
        
        matching_size = 0
        while bfs():
            for u in left_nodes:
                if pair_u[u] == -1:
                    if dfs(u):
                        matching_size += 1
        
        # Find Minimum Vertex Cover
        # Z = set of vertices reachable from unmatched U via alternating paths
        Z = set()
        queue = deque()
        for u in left_nodes:
            if pair_u[u] == -1:
                queue.append(u)
                Z.add(u)
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v in right_index:
                    # Edge u-v is unmatched?
                    # If pair_u[u] != v, it's an unmatched edge in the context of the alternating path from U.
                    # Actually, from U, we traverse unmatched edges.
                    if pair_u[u] != v:
                        if v not in Z:
                            Z.add(v)
                            # From V, traverse matched edges
                            pu = pair_v[v]
                            if pu != -1 and pu not in Z:
                                Z.add(pu)
                                queue.append(pu)
        
        # MVC = (U \ Z) U (V \ Z)
        # MIS = (U \cap Z) U (V \ Z)
        
        mis = set()
        for u in left_nodes:
            if u in Z:
                mis.add(u)
        for v in right_nodes:
            if v not in Z:
                mis.add(v)
        
        return list(mis)
    else:
        # Fallback: Greedy + Local Search
        degrees = {n: len(adj[n]) for n in nodes}
        sorted_nodes = sorted(nodes, key=lambda x: degrees[x])
        
        S = set()
        for n in sorted_nodes:
            conflict = False
            for m in S:
                if m in adj[n]:
                    conflict = True
                    break
            if not conflict:
                S.add(n)
        
        removed = [n for n in nodes if n not in S]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(removed)):
                n = removed[i]
                conflict = False
                for m in S:
                    if m in adj[n]:
                        conflict = True
                        break
                if not conflict:
                    S.add(n)
                    removed.pop(i)
                    improved = True
                    break
        
        return list(S)
