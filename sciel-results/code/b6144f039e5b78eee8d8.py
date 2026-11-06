import time
import sys
from collections import deque

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]}

def construct():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    # Get the largest component
    largest_comp = max(components, key=len)
    
    if len(largest_comp) == 0:
        return [0]
        
    # Map local indices to global for easier processing? 
    # Or just work with global indices but restrict to largest_comp set.
    comp_set = set(largest_comp)
    
    # Check if the component is a tree
    # Edges in component
    e_count = 0
    for u in largest_comp:
        for v in adj[u]:
            if v in comp_set and u < v:
                e_count += 1
                
    is_tree = (e_count == len(largest_comp) - 1)
    
    best_path = []
    
    if is_tree:
        # Find diameter of the tree
        # BFS from arbitrary node to find farthest
        def bfs_farthest(start):
            dist = {start: 0}
            parent = {start: None}
            q = deque([start])
            farthest = start
            max_d = 0
            while q:
                u = q.popleft()
                if dist[u] > max_d:
                    max_d = dist[u]
                    farthest = u
                for v in adj[u]:
                    if v in comp_set and v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
            return farthest, dist, parent
        
        u, dist_u, parent_u = bfs_farthest(largest_comp[0])
        v, dist_v, parent_v = bfs_farthest(u)
        
        # Reconstruct path from u to v
        # parent_v stores parents in the tree rooted at u
        path = []
        curr = v
        while curr is not None:
            path.append(curr)
            curr = parent_v[curr]
        path.reverse()
        best_path = path
    else:
        # Component has cycles. Try to find a long path.
        # Heuristic: Start from a leaf (degree 1 in the component) or low degree node.
        # Use DFS with backtracking and time limit.
        
        # Identify nodes in the component sorted by degree
        degs = {u: len([v for v in adj[u] if v in comp_set]) for u in largest_comp}
        start_nodes = sorted(largest_comp, key=lambda x: degs[x])
        
        best_len = 0
        deadline = time.time() + 80 # 80s limit
        
        def dfs_longest_path(start, path, visited_set):
            nonlocal best_path, best_len
            if time.time() > deadline:
                return
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            curr = path[-1]
            # Heuristic: try neighbors with lowest degree first (Warnsdorff-like)
            neighbors = [v for v in adj[curr] if v in comp_set and v not in visited_set]
            neighbors.sort(key=lambda x: degs[x])
            
            for v in neighbors:
                # Pruning: if current len + remaining possible <= best_len, skip?
                # Hard to estimate remaining possible quickly.
                # Just rely on time limit and good heuristic.
                path.append(v)
                visited_set.add(v)
                dfs_longest_path(v, path, visited_set)
                path.pop()
                visited_set.remove(v)
                if time.time() > deadline:
                    return

        # Try from a few promising start nodes
        for start in start_nodes[:10]:
            if time.time() > deadline:
                break
            dfs_longest_path(start, [start], {start})
            
        # If DFS didn't find anything (shouldn't happen), fallback to diameter of spanning tree
        if not best_path:
             # Construct a spanning tree and find diameter
             # Simple BFS tree
             parent = {start_nodes[0]: None}
             q = deque([start_nodes[0]])
             while q:
                 u = q.popleft()
                 for v in adj[u]:
                     if v in comp_set and v not in parent:
                         parent[v] = u
                         q.append(v)
             # Find farthest
             def bfs_farthest_tree(start, parent):
                 dist = {start: 0}
                 q = deque([start])
                 farthest = start
                 max_d = 0
                 while q:
                     u = q.popleft()
                     if dist[u] > max_d:
                         max_d = dist[u]
                         farthest = u
                     for v in adj[u]:
                         if v in comp_set and v not in dist:
                             dist[v] = dist[u] + 1
                             q.append(v)
                 return farthest, dist, parent # parent is from original BFS, need to trace back? 
                 # Actually, the parent dict from BFS gives the tree structure.
                 # We need to trace from farthest back to root.
                 
             # This fallback is complex, let's just use the first few steps of DFS result or a simple greedy
             # Greedy path
             path = [start_nodes[0]]
             seen = {start_nodes[0]}
             while True:
                 curr = path[-1]
                 nexts = [v for v in adj[curr] if v in comp_set and v not in seen]
                 if not nexts:
                     break
                 nexts.sort(key=lambda x: degs[x])
                 path.append(nexts[0])
                 seen.add(nexts[0])
             best_path = path

    return best_path

# To be safe, if the graph is disconnected and the largest component is small, 
# we might want to check if combining paths from different components is possible? 
# No, a path must be contiguous. So we can only pick one component.

# Let's double check the "seed" score.
# If the seed program returned [0], score is 1.
# If my program returns a path of length 20, score is 20.
# This is a huge improvement.

# One edge case: What if the largest component is just 1 node?
# Then best_path = [node].

# Let's refine the DFS pruning.
# If we are at node u, and the number of unvisited nodes in the component is R.
# If len(path) + R <= best_len, we can prune.
# We can precompute the set of all nodes in the component.
# remaining = len(comp_set) - len(visited_set)
# if len(path) + remaining <= best_len: return

# Let's add this pruning to the DFS.

def construct():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    largest_comp = max(components, key=len)
    if not largest_comp:
        return [0]
        
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    e_count = 0
    for u in largest_comp:
        for v in adj[u]:
            if v in comp_set and u < v:
                e_count += 1
                
    is_tree = (e_count == comp_size - 1)
    
    best_path = []
    
    if is_tree:
        # Diameter
        def bfs_farthest(start):
            dist = {start: 0}
            parent = {start: None}
            q = deque([start])
            farthest = start
            max_d = 0
            while q:
                u = q.popleft()
                if dist[u] > max_d:
                    max_d = dist[u]
                    farthest = u
                for v in adj[u]:
                    if v in comp_set and v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
            return farthest, dist, parent
        
        u, dist_u, parent_u = bfs_farthest(largest_comp[0])
        v, dist_v, parent_v = bfs_farthest(u)
        
        path = []
        curr = v
        while curr is not None:
            path.append(curr)
            curr = parent_v[curr]
        path.reverse()
        best_path = path
    else:
        degs = {u: len([v for v in adj[u] if v in comp_set]) for u in largest_comp}
        start_nodes = sorted(largest_comp, key=lambda x: degs[x])
        
        best_len = 0
        deadline = time.time() + 85
        
        def dfs_longest_path(start, path, visited_set):
            nonlocal best_path, best_len
            if time.time() > deadline:
                return
            
            # Pruning
            remaining = comp_size - len(visited_set)
            if len(path) + remaining <= best_len:
                return
                
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            curr = path[-1]
            neighbors = [v for v in adj[curr] if v in comp_set and v not in visited_set]
            neighbors.sort(key=lambda x: degs[x])
            
            for v in neighbors:
                path.append(v)
                visited_set.add(v)
                dfs_longest_path(v, path, visited_set)
                path.pop()
                visited_set.remove(v)
                if time.time() > deadline:
                    return

        for start in start_nodes[:20]: # Try more starts
            if time.time() > deadline:
                break
            dfs_longest_path(start, [start], {start})
            
        if not best_path:
            # Fallback: Greedy
            start = start_nodes[0]
            path = [start]
            seen = {start}
            while True:
                curr = path[-1]
                nexts = [v for v in adj[curr] if v in comp_set and v not in seen]
                if not nexts:
                    break
                nexts.sort(key=lambda x: degs[x])
                path.append(nexts[0])
                seen.add(nexts[0])
            best_path = path

    return best_path
