import time
import sys

# Increase recursion limit
sys.setrecursionlimit(10000)

# P is the pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            visited_comp[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    # The longest path must be in the largest component
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    best_path = [components[0][0]] if components else [0]
    best_len = len(best_path)
    
    largest_comp = components[0]
    comp_set = set(largest_comp)
    
    # Check if tree
    e_count = 0
    for u in largest_comp:
        for v in adj[u]:
            if v in comp_set and u < v:
                e_count += 1
                
    if e_count == len(largest_comp) - 1:
        # It's a tree! Find diameter.
        def bfs_farthest(start, comp_set):
            dist = {start: 0}
            parent = {start: None}
            queue = [start]
            farthest = start
            max_dist = 0
            while queue:
                u = queue.pop(0)
                if dist[u] > max_dist:
                    max_dist = dist[u]
                    farthest = u
                for v in adj[u]:
                    if v in comp_set and v not in dist:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
            # Reconstruct path
            path = [farthest]
            while parent[farthest] is not None:
                farthest = parent[farthest]
                path.append(farthest)
            path.reverse()
            return path
        
        start_node = largest_comp[0]
        path1 = bfs_farthest(start_node, comp_set)
        end_node_a = path1[-1]
        path2 = bfs_farthest(end_node_a, comp_set)
        
        if len(path2) > best_len:
            best_path = path2
            best_len = len(best_path)
    else:
        # Not a tree. Try to handle Unicyclic case specifically.
        # Find 2-core
        def find_2_core(nodes):
            core = set(nodes)
            deg = {u: len([v for v in adj[u] if v in core]) for u in core}
            changed = True
            while changed:
                changed = False
                for u in list(core):
                    if deg[u] < 2:
                        core.remove(u)
                        changed = True
                        for v in adj[u]:
                            if v in core:
                                deg[v] -= 1
            return core, deg
        
        core, core_deg = find_2_core(largest_comp)
        
        if len(core) == 0:
            # Should have been a tree, but maybe my e_count was wrong?
            # Fallback to DFS
            pass
        elif len(core) == 1:
            # Self-loop? Unlikely in simple graph.
            pass
        else:
            # Check if core is a simple cycle
            is_cycle = all(deg == 2 for deg in core_deg.values())
            if is_cycle:
                # It's unicyclic.
                # Find the cycle order
                cycle_nodes = list(core)
                start_c = cycle_nodes[0]
                cycle_order = [start_c]
                visited_c = {start_c}
                curr = start_c
                # Pick a neighbor in core
                next_c = [v for v in adj[curr] if v in core][0]
                curr = next_c
                while curr != start_c:
                    cycle_order.append(curr)
                    visited_c.add(curr)
                    # Find next neighbor in core not visited
                    nexts = [v for v in adj[curr] if v in core and v not in visited_c]
                    if not nexts:
                        break # Should not happen if it's a cycle
                    curr = nexts[0]
                
                # Calculate height h[u] for each u in core
                # h[u] is the length of the longest path starting at u, going into the tree attached to u (excluding cycle edges)
                # Note: The path can go "up" from u into the tree.
                # We want the max depth from u in the tree formed by removing cycle edges.
                
                def get_max_depth(start, exclude_set):
                    # BFS/DFS to find max depth in the tree attached to start
                    # exclude_set contains the cycle edges/neighbors to avoid
                    visited_d = {start}
                    max_depth = 0
                    stack = [(start, 0)]
                    while stack:
                        u, d = stack.pop()
                        if d > max_depth:
                            max_depth = d
                        for v in adj[u]:
                            if v in exclude_set:
                                continue
                            if v not in visited_d:
                                visited_d.add(v)
                                stack.append((v, d+1))
                    return max_depth

                # For each node in core, determine its "tree" neighbors (non-core neighbors)
                # And compute the max depth into that tree.
                # Actually, we need the diameter of the attached tree as well?
                # No, if the path enters the core at u and exits at v, it uses h[u] and h[v].
                # If the path stays in the tree attached to u, it's the diameter of that tree.
                
                # Let's compute h[u] = max depth from u into the attached tree.
                # And D_u = diameter of the attached tree.
                
                heights = {}
                diameters = {}
                
                for u in core:
                    # The attached tree is the component containing u in G - E(core)
                    # But we just want the depth from u.
                    # And the diameter of that tree.
                    
                    # To get diameter, we can find the two farthest nodes in the attached tree.
                    # But the attached tree is rooted at u? No, it's a tree attached at u.
                    # The diameter of the attached tree might not pass through u.
                    
                    # Let's find the nodes in the attached tree.
                    attached_nodes = set()
                    stack = [u]
                    visited_att = {u}
                    # We traverse from u, but we cannot go to core neighbors.
                    # Wait, u is in core. Its neighbors in core are the cycle neighbors.
                    # We want to traverse to neighbors NOT in core.
                    # And then their neighbors, etc.
                    
                    # Actually, the "attached tree" is the set of nodes reachable from u without using core-core edges.
                    # But u is in core. So we start at u, and go to non-core neighbors.
                    
                    # Let's just do a BFS from u, but forbidding moves to other core nodes.
                    # This will find the "branch" attached to u.
                    
                    branch_nodes = {u}
                    stack = [u]
                    visited_b = {u}
                    parent_b = {u: None}
                    
                    while stack:
                        x = stack.pop()
                        for y in adj[x]:
                            if y in core and y != u: 
                                # If y is in core and not u, we skip it.
                                # This ensures we stay in the tree attached to u.
                                # Note: If u has a neighbor y in core, we don't go there.
                                continue
                            if y not in visited_b:
                                visited_b.add(y)
                                parent_b[y] = x
                                stack.append(y)
                                branch_nodes.add(y)
                                
                    # Now we have the branch nodes.
                    # Find the diameter of this branch tree.
                    # The branch tree is a tree rooted at u?
                    # Not necessarily, but it is a tree.
                    # We can find the diameter by picking any node, finding farthest, then farthest from that.
                    
                    if len(branch_nodes) == 1:
                        diameters[u] = 0
                        heights[u] = 0
                    else:
                        # Pick u as start
                        def bfs_farthest_in_branch(start, branch_set):
                            dist = {start: 0}
                            parent = {start: None}
                            queue = [start]
                            farthest = start
                            max_dist = 0
                            while queue:
                                x = queue.pop(0)
                                if dist[x] > max_dist:
                                    max_dist = dist[x]
                                    farthest = x
                                for y in adj[x]:
                                    if y in branch_set and y not in dist:
                                        # Ensure we don't cross core edges?
                                        # The branch_set is defined such that it doesn't contain other core nodes.
                                        # So any edge in branch_set is valid.
                                        dist[y] = dist[x] + 1
                                        parent[y] = x
                                        queue.append(y)
                            path = [farthest]
                            while parent[farthest] is not None:
                                farthest = parent[farthest]
                                path.append(farthest)
                            path.reverse()
                            return farthest, path

                        f1, p1 = bfs_farthest_in_branch(u, branch_nodes)
                        f2, p2 = bfs_farthest_in_branch(f1, branch_nodes)
                        diameters[u] = len(p2) - 1
                        # Height from u:
                        # Max dist from u to any node in branch
                        # We can use the dist from the first BFS? No, that was from u to f1.
                        # We need max dist from u.
                        # Let's just run BFS from u again to get max dist.
                        dist_u = {u: 0}
                        queue = [u]
                        max_d = 0
                        while queue:
                            x = queue.pop(0)
                            if dist_u[x] > max_d:
                                max_d = dist_u[x]
                            for y in adj[x]:
                                if y in branch_nodes and y not in dist_u:
                                    dist_u[y] = dist_u[x] + 1
                                    queue.append(y)
                        heights[u] = max_d

                # Now, find the best path using the cycle.
                # Path 1: Max diameter of any attached tree.
                max_diam = max(diameters.values())
                # We need the actual path for this.
                # We'll store the best path found so far.
                
                # Path 2: Entering at i, exiting at j.
                # Length = heights[i] + dist_cycle(i, j) + heights[j]
                # dist_cycle is the number of edges on the cycle between i and j.
                # We have cycle_order.
                k = len(cycle_order)
                # Map node to index in cycle_order
                idx_map = {node: i for i, node in enumerate(cycle_order)}
                
                best_cycle_path_len = 0
                best_cycle_path = None
                
                for i in range(k):
                    u = cycle_order[i]
                    for j in range(i+1, k):
                        v = cycle_order[j]
                        # Two arcs
                        len1 = j - i
                        len2 = k - len1
                        # We can take either arc.
                        # The path is u -> ... -> v on the cycle.
                        # Plus the branches.
                        # Total length = heights[u] + len_arc + heights[v]
                        # Note: heights are number of edges.
                        # Path vertices = edges + 1.
                        
                        # Let's calculate vertex count.
                        # Branch at u has height h_u (edges). So h_u+1 vertices?
                        # No, if we enter the cycle at u, we use the path from the leaf of the branch to u.
                        # That path has h_u edges.
                        # Then we traverse the cycle arc (len_arc edges).
                        # Then we traverse the branch at v (h_v edges).
                        # Total edges = h_u + len_arc + h_v.
                        # Total vertices = h_u + len_arc + h_v + 1.
                        
                        # We want to maximize this.
                        # We need to reconstruct the path.
                        
                        # For now, just check length.
                        l1 = heights[u] + len1 + heights[v]
                        l2 = heights[u] + len2 + heights[v]
                        
                        if l1 > best_cycle_path_len:
                            best_cycle_path_len = l1
                            # Store info to reconstruct
                            # We need the path in branch u, the arc, and path in branch v.
                            # We'll reconstruct later.
                            best_cycle_info = (u, v, len1, 1) # 1 for arc 1
                        if l2 > best_cycle_path_len:
                            best_cycle_path_len = l2
                            best_cycle_info = (u, v, len2, 2) # 2 for arc 2
                
                # Compare with max_diam
                # If best_cycle_path_len + 1 > max_diam + 1, use cycle path.
                # Note: max_diam is edges. Vertices = max_diam + 1.
                
                # We need to reconstruct the best path.
                # This requires storing the actual paths for heights and diameters.
                # To save code, let's just run the DFS if the cycle path is promising,
                # or implement the reconstruction.
                
                # Given time, let's just run a targeted DFS from the endpoints of the best cycle path?
                # Or just return the best of the diameters if we don't have the path.
                
                # Actually, let's just use the DFS for non-tree cases but with a time limit,
                # and hope it finds a good path. The unicyclic logic is complex to implement fully with path reconstruction in a short edit.
                
                # However, if we know the max possible length, we can guide the DFS.
                
                pass # Fallback to DFS below

        # Fallback DFS for non-tree cases
        degree = [len(adj[i]) for i in range(n)]
        
        def count_reachable(current, visited_set):
            bstack = [current]
            bseen = {current}
            count = 0
            while bstack:
                u = bstack.pop()
                for v in adj[u]:
                    if v in visited_set:
                        continue
                    if v not in bseen:
                        bseen.add(v)
                        count += 1
                        bstack.append(v)
            return count
        
        def dfs(path, current, visited):
            nonlocal best_path, best_len
            
            if len(path) > best_len:
                best_path = path[:]
                best_len = len(best_path)
            
            total_unvisited = n - len(visited)
            if len(path) + total_unvisited <= best_len:
                return
            
            neighbors = [v for v in adj[current] if v not in visited]
            
            if not neighbors:
                return
            
            if len(neighbors) > 1 or len(path) > best_len - 10:
                reachable_count = count_reachable(current, visited)
                if len(path) + reachable_count <= best_len:
                    return
            
            def heuristic(v):
                unvis_nei = 0
                for w in adj[v]:
                    if w not in visited:
                        unvis_nei += 1
                return unvis_nei
            
            neighbors.sort(key=heuristic)
            
            for v in neighbors:
                if time.time() >= deadline:
                    return
                path.append(v)
                visited.add(v)
                dfs(path, v, visited)
                path.pop()
                visited.discard(v)
                if time.time() >= deadline:
                    return
        
        sorted_starts = sorted(largest_comp, key=lambda x: degree[x])
        for start in sorted_starts:
            if time.time() >= deadline:
                break
            path = [start]
            visited = {start}
            dfs(path, start, visited)
            if time.time() >= deadline:
                break

    return best_path

# EVOLVE-BLOCK-END
