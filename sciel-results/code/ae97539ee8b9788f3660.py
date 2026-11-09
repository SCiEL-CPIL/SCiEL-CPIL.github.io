"""Rewritten program: Longest path via Tree Diameter + Local Search for Cyclic Components."""
import random
from collections import deque

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        queue.append(nei)
            components.append(comp)
            
    # Find the largest component
    if not components:
        return [0]
    largest_comp = max(components, key=len)
    
    # Check if the largest component is a tree
    num_vertices = len(largest_comp)
    num_edges_in_comp = 0
    for v in largest_comp:
        num_edges_in_comp += len([nei for nei in adj[v] if nei in set(largest_comp)])
    num_edges_in_comp //= 2
    
    is_tree = (num_edges_in_comp == num_vertices - 1)
    
    def find_diameter(vertices):
        """Find the diameter of a tree given its vertices."""
        if not vertices:
            return []
        if len(vertices) == 1:
            return vertices
        
        # BFS to find farthest node from an arbitrary node
        def bfs_farthest(start):
            dist = {start: 0}
            parent = {start: None}
            queue = deque([start])
            farthest = start
            while queue:
                node = queue.popleft()
                for nei in adj[node]:
                    if nei in set(vertices) and nei not in dist:
                        dist[nei] = dist[node] + 1
                        parent[nei] = node
                        queue.append(nei)
                        if dist[nei] > dist[farthest]:
                            farthest = nei
            return farthest, parent
        
        # Step 1: Pick arbitrary node, find farthest
        start = vertices[0]
        far1, parent1 = bfs_farthest(start)
        
        # Step 2: From far1, find farthest
        far2, parent2 = bfs_farthest(far1)
        
        # Reconstruct path
        path = []
        curr = far2
        while curr is not None:
            path.append(curr)
            curr = parent2[curr]
        path.reverse()
        return path

    def local_search(path):
        """Improve path using 2-Opt and Relocate moves."""
        path = list(path)
        n_path = len(path)
        if n_path < 3:
            return path
        
        improved = True
        max_iterations = 100
        iter_count = 0
        
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            n_path = len(path)
            
            # Try 2-Opt moves
            # Path: ... a - b ... c - d ...
            # Break b-c and a-d? No, 2-opt in path:
            # Reverse segment [i..j].
            # New path: ... path[i-1] - path[j] - ... - path[i] - path[j+1] ...
            # We need edge (path[i-1], path[j]) and (path[i], path[j+1])
            
            for i in range(1, n_path - 1):
                for j in range(i + 1, n_path):
                    # Check if reversing path[i..j] is valid
                    # Current: ... path[i-1] - path[i] ... path[j] - path[j+1] ...
                    # New: ... path[i-1] - path[j] ... path[i] - path[j+1] ...
                    # Need edges: (path[i-1], path[j]) and (path[i], path[j+1])
                    if j == n_path - 1:
                        # Reversing to the end
                        # New end is path[i]
                        # Need edge (path[i-1], path[j])
                        if path[j] in adj[path[i-1]]:
                            # Construct new path
                            new_path = path[:i] + path[i:j+1][::-1]
                            # Check if valid
                            valid = True
                            for k in range(len(new_path) - 1):
                                if new_path[k+1] not in adj[new_path[k]]:
                                    valid = False
                                    break
                            if valid:
                                # Try to extend
                                new_path = extend_path(new_path)
                                if len(new_path) > len(path):
                                    path = new_path
                                    improved = True
                                    break
                    else:
                        if path[j] in adj[path[i-1]] and path[i] in adj[path[j+1]]:
                            new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                            # Check validity
                            valid = True
                            for k in range(len(new_path) - 1):
                                if new_path[k+1] not in adj[new_path[k]]:
                                    valid = False
                                    break
                            if valid:
                                new_path = extend_path(new_path)
                                if len(new_path) > len(path):
                                    path = new_path
                                    improved = True
                                    break
                if improved:
                    break
            
            # Try Relocate moves
            # Move vertex at i to position j
            if not improved:
                for i in range(1, len(path) - 1): # Don't move ends initially to keep it simple, or allow
                    v = path[i]
                    # Try inserting v after each other vertex
                    for j in range(len(path)):
                        if j == i or j == i - 1:
                            continue
                        # Remove v
                        path_without = path[:i] + path[i+1:]
                        # Try inserting at j
                        # Insert before path_without[j]
                        insert_pos = j
                        # Check edges: (path_without[insert_pos-1], v) and (v, path_without[insert_pos])
                        if insert_pos == 0:
                            if v in adj[path_without[0]]:
                                new_path = [v] + path_without
                                new_path = extend_path(new_path)
                                if len(new_path) > len(path):
                                    path = new_path
                                    improved = True
                                    break
                        elif insert_pos == len(path_without):
                            if v in adj[path_without[-1]]:
                                new_path = path_without + [v]
                                new_path = extend_path(new_path)
                                if len(new_path) > len(path):
                                    path = new_path
                                    improved = True
                                    break
                        else:
                            if v in adj[path_without[insert_pos-1]] and path_without[insert_pos] in adj[v]:
                                new_path = path_without[:insert_pos] + [v] + path_without[insert_pos:]
                                new_path = extend_path(new_path)
                                if len(new_path) > len(path):
                                    path = new_path
                                    improved = True
                                    break
                    if improved:
                        break
                        
        return path

    def extend_path(path):
        """Greedy extension of path at both ends."""
        path = list(path)
        changed = True
        while changed:
            changed = False
            path_set = set(path)
            
            # Extend front
            front = path[0]
            cands = [v for v in adj[front] if v not in path_set]
            if cands:
                # Pick candidate that leads to longest greedy extension
                best_v = None
                best_len = 0
                for v in cands:
                    # Simple greedy from v
                    ext = [v]
                    seen = path_set | {v}
                    while True:
                        next_cands = [w for w in adj[ext[-1]] if w not in seen]
                        if not next_cands:
                            break
                        # Warnsdorff
                        w = min(next_cands, key=lambda x: (len(adj[x] - seen), x))
                        ext.append(w)
                        seen.add(w)
                    if len(ext) > best_len:
                        best_len = len(ext)
                        best_v = v
                if best_v is not None:
                    path.insert(0, best_v)
                    changed = True
            
            # Extend back
            back = path[-1]
            path_set = set(path)
            cands = [v for v in adj[back] if v not in path_set]
            if cands:
                best_v = None
                best_len = 0
                for v in cands:
                    ext = [v]
                    seen = path_set | {v}
                    while True:
                        next_cands = [w for w in adj[ext[-1]] if w not in seen]
                        if not next_cands:
                            break
                        w = min(next_cands, key=lambda x: (len(adj[x] - seen), x))
                        ext.append(w)
                        seen.add(w)
                    if len(ext) > best_len:
                        best_len = len(ext)
                        best_v = v
                if best_v is not None:
                    path.append(best_v)
                    changed = True
                    
        return path

    if is_tree:
        best_path = find_diameter(largest_comp)
    else:
        # Use diameter of a spanning tree as seed
        # Construct a spanning tree using BFS
        start = largest_comp[0]
        visited = {start}
        parent = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nei in adj[node]:
                if nei in set(largest_comp) and nei not in visited:
                    visited.add(nei)
                    parent[nei] = node
                    queue.append(nei)
        
        # Find diameter of this tree
        # We can reuse find_diameter logic but on the tree edges
        # For simplicity, just use greedy path on the component as seed
        # Or use the diameter of the spanning tree
        # Let's find the two farthest nodes in the spanning tree
        def bfs_tree_farthest(start, tree_adj):
            dist = {start: 0}
            parent = {start: None}
            queue = deque([start])
            farthest = start
            while queue:
                node = queue.popleft()
                for nei in tree_adj[node]:
                    if nei not in dist:
                        dist[nei] = dist[node] + 1
                        parent[nei] = node
                        queue.append(nei)
                        if dist[nei] > dist[farthest]:
                            farthest = nei
            return farthest, parent

        # Build tree adjacency
        tree_adj = {v: [] for v in largest_comp}
        for v in largest_comp:
            if parent[v] is not None:
                tree_adj[v].append(parent[v])
                tree_adj[parent[v]].append(v)
        
        far1, p1 = bfs_tree_farthest(start, tree_adj)
        far2, p2 = bfs_tree_farthest(far1, tree_adj)
        
        # Reconstruct path
        path = []
        curr = far2
        while curr is not None:
            path.append(curr)
            curr = p2[curr]
        path.reverse()
        
        best_path = path

    # Local Search
    best_path = local_search(best_path)
    
    # Final Extension
    best_path = extend_path(best_path)
    
    # Verify
    seen = set()
    valid_path = []
    for v in best_path:
        if v not in seen:
            valid_path.append(v)
            seen.add(v)
    final_path = [valid_path[0]]
    for i in range(1, len(valid_path)):
        if valid_path[i] in adj[final_path[-1]]:
            final_path.append(valid_path[i])
            
    if not final_path:
        final_path = [0]
        
    return final_path
