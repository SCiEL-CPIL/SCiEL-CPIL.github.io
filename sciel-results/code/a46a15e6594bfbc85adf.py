import time
import collections
import random
import heapq

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
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
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(comp)
    
    best_path = []
    
    # Helper to extend a path greedily
    def greedy_extend(path, adj, visited_set):
        """Extend path from both ends greedily using a priority queue."""
        if not path:
            return path
        
        path = list(path)
        visited = set(path)
        
        # We'll extend from both ends
        # Use a list to represent the path, and extend from left or right
        
        changed = True
        while changed:
            changed = False
            # Try to extend from the left end
            left_end = path[0]
            right_end = path[-1]
            
            best_left = None
            best_right = None
            
            # Score neighbors: prefer those with fewer unvisited neighbors (to avoid dead ends)
            # Or prefer those with more unvisited neighbors (to keep options open)?
            # Usually, for longest path, "degree" heuristic works: pick neighbor with lowest degree among unvisited.
            
            unvisited_left = [nb for nb in adj[left_end] if nb not in visited]
            unvisited_right = [nb for nb in adj[right_end] if nb not in visited]
            
            if unvisited_left:
                # Pick the one with minimum degree
                best_left = min(unvisited_left, key=lambda x: len(adj[x]))
            
            if unvisited_right:
                best_right = min(unvisited_right, key=lambda x: len(adj[x]))
            
            # Decide which side to extend
            # If both available, pick the one that leads to a node with lower degree?
            # Or just alternate?
            # Let's try: if left is available, extend left. Else right.
            # Actually, a better heuristic: extend the side where the candidate has the lowest degree.
            
            if best_left is not None and best_right is not None:
                # Compare degrees of the candidates
                if len(adj[best_left]) <= len(adj[best_right]):
                    path.insert(0, best_left)
                    visited.add(best_left)
                    changed = True
                else:
                    path.append(best_right)
                    visited.add(best_right)
                    changed = True
            elif best_left is not None:
                path.insert(0, best_left)
                visited.add(best_left)
                changed = True
            elif best_right is not None:
                path.append(best_right)
                visited.add(best_right)
                changed = True
        
        return path
    
    # Helper to try local search improvements (2-opt style)
    def local_search(path, adj, deadline):
        """Try to improve path by reordering or extending."""
        if not path or len(path) < 2:
            return path
        
        path = list(path)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_path = len(path)
            
            # Try 2-opt: reverse a segment
            # A 2-opt move in a path: pick i, j (i < j-1), reverse path[i:j+1]
            # This is valid if edge (path[i-1], path[j]) and (path[i], path[j+1]) exist (with boundaries handled)
            # Wait, reversing a segment in a path changes the connections at the boundaries.
            # Original: ... A - B ... C - D ...
            # Reverse B..C: ... A - C ... B - D ...
            # Need edge (A, C) and (B, D).
            
            found_move = False
            for i in range(n_path):
                if time.time() > deadline:
                    break
                for j in range(i + 2, n_path):
                    if time.time() > deadline:
                        break
                    
                    # Check boundaries
                    # Left boundary: if i > 0, need edge (path[i-1], path[j])
                    # Right boundary: if j < n_path - 1, need edge (path[i], path[j+1])
                    
                    left_ok = True
                    if i > 0:
                        if path[i-1] not in adj[path[j]]:
                            left_ok = False
                    
                    right_ok = True
                    if j < n_path - 1:
                        if path[i] not in adj[path[j+1]]:
                            right_ok = False
                    
                    if left_ok and right_ok:
                        # Perform reversal
                        new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                        # Verify validity
                        valid = True
                        for k in range(len(new_path) - 1):
                            if new_path[k] not in adj[new_path[k+1]]:
                                valid = False
                                break
                        if valid:
                            path = new_path
                            improved = True
                            found_move = True
                            break
                if found_move:
                    break
            
            # Try extending if possible
            if not improved:
                extended = greedy_extend(path, adj, set())
                if len(extended) > len(path):
                    path = extended
                    improved = True
        
        return path
    
    # Process each component
    for comp in components:
        if len(comp) <= len(best_path):
            continue
        
        comp_set = set(comp)
        
        # Strategy 1: Tree diameter (if it's a tree)
        edge_count = sum(len(adj[u]) for u in comp) // 2
        if edge_count == len(comp) - 1:
            # It's a tree
            # Find diameter
            def bfs(start, comp_set, adj):
                dist = {start: 0}
                parent = {start: None}
                q = collections.deque([start])
                farthest = start
                max_dist = 0
                while q:
                    u = q.popleft()
                    if dist[u] > max_dist:
                        max_dist = dist[u]
                        farthest = u
                    for v in adj[u]:
                        if v in comp_set and v not in dist:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                return farthest, parent, dist
            
            u, parent_u, _ = bfs(comp[0], comp_set, adj)
            v, parent_v, _ = bfs(u, comp_set, adj)
            
            # Reconstruct path
            path = []
            curr = v
            while curr is not None:
                path.append(curr)
                if curr == u:
                    break
                curr = parent_v[curr]
            path.reverse()
            
            if len(path) > len(best_path):
                best_path = path
        
        # Strategy 2: Greedy + Local Search
        # Try multiple random starts
        rng = random.Random(42)
        candidates = []
        
        # Add some deterministic starts
        starts = list(comp)
        rng.shuffle(starts)
        
        # Limit number of starts based on time
        max_starts = 20
        for start in starts[:max_starts]:
            if time.time() > deadline:
                break
            
            # Start with just the node
            path = [start]
            
            # Greedy extend
            path = greedy_extend(path, adj, set())
            
            # Local search
            path = local_search(path, adj, deadline)
            
            candidates.append(path)
            
            if len(path) > len(best_path):
                best_path = path
        
        # Also try starting from edges
        for u, v in P['edges']:
            if u in comp_set and v in comp_set:
                if time.time() > deadline:
                    break
                
                path = [u, v]
                path = greedy_extend(path, adj, set())
                path = local_search(path, adj, deadline)
                
                candidates.append(path)
                
                if len(path) > len(best_path):
                    best_path = path
        
        # Try combining? No, simple path must be simple.
        
        # Final local search on best candidate from this component
        if candidates:
            best_comp_path = max(candidates, key=len)
            best_comp_path = local_search(best_comp_path, adj, deadline)
            if len(best_comp_path) > len(best_path):
                best_path = best_comp_path
    
    # Ensure best_path is not empty
    if not best_path:
        # Fallback: pick any edge
        for u, v in P['edges']:
            best_path = [u, v]
            break
    
    return best_path
# EVOLVE-BLOCK-END
