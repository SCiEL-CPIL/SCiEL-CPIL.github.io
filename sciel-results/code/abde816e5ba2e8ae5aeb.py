import time
import sys
from collections import deque

# Increase recursion limit
sys.setrecursionlimit(10000)

# P is the pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    deadline = time.time() + 80  # Safety margin
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # 1. Find 2-core
    deg = [len(adj[i]) for i in range(n)]
    in_core = [False] * n
    removed = [False] * n
    
    # Iteratively remove nodes with degree < 2
    # Use a queue for efficiency
    q = deque([i for i in range(n) if deg[i] < 2])
    for i in q:
        removed[i] = True
        
    while q:
        u = q.popleft()
        if not removed[u]: continue
        for v in adj[u]:
            if not removed[v]:
                deg[v] -= 1
                if deg[v] < 2:
                    removed[v] = True
                    q.append(v)
                    
    core_nodes = [i for i in range(n) if not removed[i]]
    
    # 2. Calculate heights h1(v) and h2(v) for core nodes
    # h1(v) is the length of the longest path starting from v into non-core nodes
    # h2(v) is the length of the second longest such path
    # Note: The path length in terms of vertices.
    # If v has no non-core neighbors, h1=0, h2=0.
    # If v has one non-core neighbor u, and u has height H, then h1(v) = H + 1 (counting u? No, counting vertices in the tree part).
    # Let's define height(v) as the number of vertices in the longest path starting at v and going into the tree.
    # If v is in core, and neighbor u is in tree:
    # Path is v -> u -> ...
    # Length added to the core path is the number of tree vertices visited.
    
    h1 = [0] * n
    h2 = [0] * n
    
    # We need to process the tree parts.
    # The tree parts are the components of the graph induced by removed nodes, attached to core nodes.
    # We can do a DFS from each core node into the removed nodes.
    
    def calc_heights(start_core_node):
        # DFS from start_core_node into removed nodes
        # Returns list of heights of branches
        # We want the top 2 heights.
        heights = []
        
        def dfs_tree(u, parent, depth):
            # u is a removed node
            max_depth = depth
            for v in adj[u]:
                if v == parent: continue
                if removed[v]:
                    d = dfs_tree(v, u, depth + 1)
                    if d > max_depth:
                        max_depth = d
                # if v is core, we don't go there (we stop at the boundary)
            return max_depth
            
        for v in adj[start_core_node]:
            if removed[v]:
                # This is a branch
                # The path starts at start_core_node, goes to v.
                # The depth of v is 1 (counting v).
                # dfs_tree returns the max depth from v.
                # Let's say dfs_tree(v, start_core_node, 1) returns max_depth.
                # This max_depth includes v.
                h = dfs_tree(v, start_core_node, 1)
                heights.append(h)
                
        heights.sort(reverse=True)
        if len(heights) >= 1:
            h1[start_core_node] = heights[0]
        if len(heights) >= 2:
            h2[start_core_node] = heights[1]
            
    for v in core_nodes:
        calc_heights(v)
        
    # 3. Find best path
    best_path = []
    
    if not core_nodes:
        # Graph is a forest
        # Find max diameter in any tree
        # Since all nodes are removed, we just find components of the original graph
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                # BFS to find component
                comp = []
                queue = [i]
                visited[i] = True
                while queue:
                    u = queue.pop(0)
                    comp.append(u)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                # Find diameter of this tree
                if len(comp) > 1:
                    # Pick arbitrary node, find farthest
                    start = comp[0]
                    dist = {start: 0}
                    parent = {start: None}
                    q2 = [start]
                    farthest = start
                    max_d = 0
                    while q2:
                        u = q2.pop(0)
                        if dist[u] > max_d:
                            max_d = dist[u]
                            farthest = u
                        for v in adj[u]:
                            if v in comp and v not in dist:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                q2.append(v)
                    
                    # Find farthest from farthest
                    dist2 = {farthest: 0}
                    parent2 = {farthest: None}
                    q3 = [farthest]
                    farthest2 = farthest
                    max_d2 = 0
                    while q3:
                        u = q3.pop(0)
                        if dist2[u] > max_d2:
                            max_d2 = dist2[u]
                            farthest2 = u
                        for v in adj[u]:
                            if v in comp and v not in dist2:
                                dist2[v] = dist2[u] + 1
                                parent2[v] = u
                                q3.append(v)
                    
                    # Reconstruct path
                    path = [farthest2]
                    while parent2[farthest2] is not None:
                        farthest2 = parent2[farthest2]
                        path.append(farthest2)
                    path.reverse()
                    
                    if len(path) > len(best_path):
                        best_path = path
                else:
                    if len(best_path) < 1:
                        best_path = [comp[0]]
                        
        return best_path

    # Core is not empty
    # We need to find a simple path in the core that maximizes:
    # h1(start) + len(path) + h1(end)
    # Special case: len(path) == 1, score = h1(v) + 1 + h2(v)
    
    # If core is small, enumerate all simple paths
    # If core is large, we might need a heuristic, but let's try enumeration first.
    # For G(100, 0.018), core is likely small.
    
    if len(core_nodes) <= 20:
        # Enumerate all simple paths in the core
        # To speed up, we can use DFS from each node
        best_score = -1
        best_core_path = None
        
        for start in core_nodes:
            # DFS
            path = [start]
            visited_core = {start}
            
            def dfs_core():
                nonlocal best_score, best_core_path
                # Calculate score for current path
                # If len(path) == 1:
                #   score = h1(start) + 1 + h2(start)
                # Else:
                #   score = h1(start) + len(path) + h1(path[-1])
                
                end = path[-1]
                if len(path) == 1:
                    score = h1[start] + 1 + h2[start]
                else:
                    score = h1[start] + len(path) + h1[end]
                
                if score > best_score:
                    best_score = score
                    best_core_path = path[:]
                    
                # Extend
                for v in adj[end]:
                    if v in core_nodes and v not in visited_core:
                        visited_core.add(v)
                        path.append(v)
                        dfs_core()
                        path.pop()
                        visited_core.discard(v)
                        
            dfs_core()
            
        if best_core_path:
            # Reconstruct full path
            start = best_core_path[0]
            end = best_core_path[-1]
            
            # Get the tree path for start
            def get_tree_path(core_node, other_core_node=None):
                # Find the longest path from core_node into the tree
                # We need to actually construct the path, not just the length.
                # This requires re-doing the DFS with path tracking.
                # For efficiency, we can store the parent pointers during height calculation.
                # But we didn't. So we do it again.
                
                best_p = []
                
                def dfs_tree_path(u, parent, current_path):
                    nonlocal best_p
                    if len(current_path) > len(best_p):
                        best_p = current_path[:]
                    
                    for v in adj[u]:
                        if v == parent: continue
                        if removed[v]:
                            dfs_tree_path(v, u, current_path + [v])
                        # If v is core and v != other_core_node, we don't go there (unless it's the other end, but we stop at core)
                        # Actually, if v is core, we stop.
                        
                # We start from core_node, go to a removed neighbor
                for v in adj[core_node]:
                    if removed[v]:
                        dfs_tree_path(v, core_node, [v])
                        
                return best_p
                
            # Note: get_tree_path returns the path of tree vertices.
            # We need to prepend core_node and append core_node.
            
            start_tree = get_tree_path(start)
            end_tree = get_tree_path(end)
            
            # If start == end (len(core_path)==1)
            if start == end:
                # We need two disjoint tree paths from start
                # My get_tree_path only finds one.
                # We need to find the top 2 disjoint paths.
                # This is more complex.
                # For now, if core path is 1 node, we might miss the optimal if it uses two branches.
                # But let's handle the general case first.
                
                # To handle start==end correctly:
                # Find two longest paths from start into the tree that are vertex-disjoint (except at start).
                # Since they are trees, any two paths from start are disjoint except at start.
                # So we just need the top 2 heights.
                # But we need the actual paths.
                
                # Let's modify get_tree_path to return the top 2.
                
                def get_two_tree_paths(core_node):
                    # Returns two longest paths from core_node into removed nodes
                    paths = []
                    
                    def dfs_tree_path(u, parent, current_path):
                        nonlocal paths
                        if len(current_path) > 0:
                            # We want to collect the best ones
                            # This DFS is exponential if not careful, but trees are small
                            # Actually, we just want the longest.
                            # We can do a DFS that returns the longest path.
                            pass
                    
                    # Simpler: Just find the longest path, remove those nodes, find the next longest?
                    # Or just find all leaves and compute distances?
                    
                    # Since the "tree" attached to core_node is a tree, the longest path from core_node is simply the height.
                    # And the second longest is the second height.
                    # We can find them by doing a DFS that returns the top 2 heights and the corresponding paths.
                    
                    def dfs_top2(u, parent):
                        # Returns (max1, path1, max2, path2)
                        m1 = 0
                        p1 = []
                        m2 = 0
                        p2 = []
                        
                        for v in adj[u]:
                            if v == parent: continue
                            if removed[v]:
                                r1, rp1, r2, rp2 = dfs_top2(v, u)
                                # The path from u through v is [v] + rp1
                                # Length is 1 + r1
                                l1 = 1 + r1
                                lp1 = [v] + rp1
                                l2 = 1 + r2
                                lp2 = [v] + rp2
                                
                                # Update m1, m2
                                # Candidates: (l1, lp1), (l2, lp2), (m1, p1), (m2, p2)
                                # Sort by length
                                candidates = []
                                if l1 > 0: candidates.append((l1, lp1))
                                if l2 > 0: candidates.append((l2, lp2))
                                if m1 > 0: candidates.append((m1, p1))
                                if m2 > 0: candidates.append((m2, p2))
                                
                                candidates.sort(key=lambda x: x[0], reverse=True)
                                
                                if len(candidates) >= 1:
                                    m1 = candidates[0][0]
                                    p1 = candidates[0][1]
                                if len(candidates) >= 2:
                                    m2 = candidates[1][0]
                                    p2 = candidates[1][1]
                                    
                        return m1, p1, m2, p2
                        
                    m1, p1, m2, p2 = dfs_top2(core_node, -1)
                    return p1, p2
                
                p1, p2 = get_two_tree_paths(start)
                full_path = p1 + [start] + list(reversed(p2))
                if len(full_path) > len(best_path):
                    best_path = full_path
            else:
                full_path = start_tree + best_core_path + list(reversed(end_tree))
                if len(full_path) > len(best_path):
                    best_path = full_path
    
    # Fallback: If core is large or enumeration failed, use a greedy heuristic on the whole graph
    # But for this problem, core is small.
    
    if not best_path:
        # Just return a single node
        best_path = [0]
        
    return best_path

# EVOLVE-BLOCK-END
