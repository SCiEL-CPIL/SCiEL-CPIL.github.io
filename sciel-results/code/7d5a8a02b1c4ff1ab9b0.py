"""Long path finder using Greedy Construction and Local Search (Insertion)."""

import time
import random

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_largest_component(adj, n):
    visited = [False] * n
    comp_sizes = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            size = 0
            while stack:
                u = stack.pop()
                size += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comp_sizes.append(size)
    if not comp_sizes:
        return []
    max_size = max(comp_sizes)
    # Find the vertices in the largest component
    # Re-do traversal to get vertices
    visited = [False] * n
    largest_comp = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            if len(comp) == max_size:
                largest_comp = comp
                break
    return largest_comp

def construct():
    adj = _build_adjacency()
    n = P['n']
    
    # Find the largest connected component to focus our search
    comp_vertices = _find_largest_component(adj, n)
    if not comp_vertices:
        return [0] # Fallback
    
    comp_set = set(comp_vertices)
    
    best_path = []
    best_len = 0
    
    start_time = time.time()
    time_limit = 55.0
    
    # Greedy + Local Search
    # We will try multiple random starts and greedy strategies
    
    def greedy_construct(start_node, adj, comp_set):
        """
        Constructs a path starting from start_node using a greedy heuristic.
        Heuristic: Pick neighbor with fewest unvisited neighbors (Warnsdorff).
        """
        path = [start_node]
        visited = {start_node}
        
        while True:
            extended = False
            # Try to extend from the end
            end = path[-1]
            candidates = [v for v in adj[end] if v in comp_set and v not in visited]
            if candidates:
                # Sort by number of unvisited neighbors
                candidates.sort(key=lambda v: len(adj[v] & comp_set - visited))
                next_node = candidates[0]
                path.append(next_node)
                visited.add(next_node)
                extended = True
            
            # Try to extend from the start
            if not extended:
                start = path[0]
                candidates = [v for v in adj[start] if v in comp_set and v not in visited]
                if candidates:
                    candidates.sort(key=lambda v: len(adj[v] & comp_set - visited))
                    next_node = candidates[0]
                    path.insert(0, next_node)
                    visited.add(next_node)
                    extended = True
            
            if not extended:
                break
            if time.time() - start_time > time_limit:
                break
                
        return path

    def local_search(path, adj, comp_set):
        """
        Improves the path by inserting unvisited vertices if possible.
        Returns the improved path.
        """
        path = path[:]
        visited = set(path)
        improved = True
        while improved:
            improved = False
            # Try to insert any unvisited vertex u into the path
            # u must be connected to two consecutive vertices in the path
            unvisited = [v for v in comp_set if v not in visited]
            
            # For efficiency, check edges from unvisited vertices
            # If u is unvisited, check its neighbors in the path
            for u in unvisited:
                if u in visited: continue
                neighs_in_path = [v for v in adj[u] if v in visited]
                if len(neighs_in_path) < 2:
                    continue
                
                # Check if any two neighbors in the path are consecutive
                for i in range(len(path) - 1):
                    v1 = path[i]
                    v2 = path[i+1]
                    if (v1 in neighs_in_path and v2 in neighs_in_path):
                        # Insert u between v1 and v2
                        # path: ... v1, v2 ... -> ... v1, u, v2 ...
                        path.insert(i+1, u)
                        visited.add(u)
                        improved = True
                        break
                if improved:
                    break
            
            # Also try to extend ends
            if not improved:
                # Extend start
                start = path[0]
                cands = [v for v in adj[start] if v in comp_set and v not in visited]
                if cands:
                    path.insert(0, cands[0])
                    visited.add(cands[0])
                    improved = True
                else:
                    # Extend end
                    end = path[-1]
                    cands = [v for v in adj[end] if v in comp_set and v not in visited]
                    if cands:
                        path.append(cands[0])
                        visited.add(cands[0])
                        improved = True
                        
        return path

    # Strategy:
    # 1. Try greedy from each vertex in the component (or a subset)
    # 2. Apply local search to the result
    # 3. Keep the best
    
    # To save time, we can sort vertices by degree and start with low degree ones
    # But greedy from any node might work. Let's try a few random starts and low-degree starts.
    
    vertices_sorted = sorted(comp_vertices, key=lambda x: len(adj[x]))
    
    # Try first 20 low-degree vertices as starts
    starts_to_try = vertices_sorted[:20]
    
    # Also add some random starts
    random.seed(748) # Fixed seed for reproducibility of the "random" part? 
    # Actually, the graph is fixed. The randomness is in the algorithm.
    # We should probably use a fixed seed for the algorithm's randomness to be deterministic?
    # The task says "random.Random(748)" is used for the graph generation.
    # Our algorithm can be deterministic or use its own random.
    # Let's use a fixed seed for our internal randomness to ensure consistent results.
    rng = random.Random(748)
    
    for _ in range(10):
        if comp_vertices:
            starts_to_try.append(rng.choice(comp_vertices))
            
    for s in starts_to_try:
        if time.time() - start_time > time_limit:
            break
            
        p = greedy_construct(s, adj, comp_set)
        p = local_search(p, adj, comp_set)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]
            
        # If we found a path covering the whole component, we can stop
        if best_len == len(comp_vertices):
            break

    # If best_path is empty (should not happen), return a single vertex
    if not best_path:
        best_path = [comp_vertices[0]] if comp_vertices else [0]
        
    return best_path
