"""Rewritten program: Hybrid metaheuristic with 2-opt and iterative extension for longest path."""
import time
import sys
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Identify largest connected component to focus search
    visited_comp = [False] * n
    comp_sizes = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            visited_comp[i] = True
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                for nb in adj[node]:
                    if not visited_comp[nb]:
                        visited_comp[nb] = True
                        stack.append(nb)
            comp_sizes.append((size, i))
            
    if not comp_sizes:
        return [0]
        
    comp_sizes.sort(reverse=True)
    largest_comp_size, start_node = comp_sizes[0]
    
    # Get nodes in largest component
    visited_comp = [False] * n
    largest_comp_nodes = set()
    stack = [start_node]
    visited_comp[start_node] = True
    while stack:
        node = stack.pop()
        largest_comp_nodes.add(node)
        for nb in adj[node]:
            if not visited_comp[nb]:
                visited_comp[nb] = True
                stack.append(nb)
                
    # Filter adjacency to only include nodes in the largest component
    # This speeds up checks and prevents accidental moves to other components
    adj_filtered = [set() for _ in range(n)]
    for u in largest_comp_nodes:
        for v in adj[u]:
            if v in largest_comp_nodes:
                adj_filtered[u].add(v)
                
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    best_path = []
    best_len = 0
    
    # Helper to check if path is valid (should always be true if constructed correctly, but good for safety)
    def is_valid_path(path):
        if not path:
            return False
        if len(set(path)) != len(path):
            return False
        for i in range(len(path) - 1):
            if path[i+1] not in adj_filtered[path[i]]:
                return False
        return True
        
    # Helper to extend path from both ends
    def extend_path(path):
        extended = True
        while extended:
            extended = False
            # Try extending from start
            if path:
                start_node = path[0]
                for nb in adj_filtered[start_node]:
                    if nb not in path:
                        path.insert(0, nb)
                        extended = True
                        break
            # Try extending from end
            if path:
                end_node = path[-1]
                for nb in adj_filtered[end_node]:
                    if nb not in path:
                        path.append(nb)
                        extended = True
                        break
        return path

    # Helper to perform 2-opt moves
    def two_opt(path, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            L = len(path)
            # Try all pairs (i, j) with i < j
            for i in range(L - 1):
                if time.time() > time_limit:
                    return path
                for j in range(i + 2, L):
                    # Check if reversing path[i+1..j] creates valid edges
                    # New edges: (path[i], path[j]) and (path[i+1], path[j+1] if j < L-1 else None)
                    # Old edges: (path[i], path[i+1]) and (path[j], path[j+1] if j < L-1 else None)
                    
                    # We want to replace edges (i, i+1) and (j, j+1) with (i, j) and (i+1, j+1)
                    # Wait, 2-opt on a path reverses the segment between i+1 and j.
                    # Original: ... A - B ... C - D ...
                    # Reverse B..C: ... A - C ... B - D ...
                    # Edges removed: (A,B), (C,D)
                    # Edges added: (A,C), (B,D)
                    
                    u, v = path[i], path[i+1]
                    w, z = path[j], path[j+1] if j < L-1 else None
                    
                    # If z is None, we are reversing to the end.
                    # Edge (w, z) doesn't exist, so we only remove (u, v) and (w, None) -> invalid logic.
                    # Let's stick to standard 2-opt: reverse segment [i+1, j].
                    # This changes edges:
                    # 1. Edge (path[i], path[i+1]) becomes (path[i], path[j])
                    # 2. Edge (path[j], path[j+1]) becomes (path[i+1], path[j+1]) -- ONLY IF j < L-1
                    #    If j == L-1, the edge (path[j], path[j+1]) doesn't exist.
                    #    The new end is path[i+1]. The edge before it is (path[i], path[j]).
                    #    So we remove (path[i], path[i+1]) and (path[j-1], path[j])? No.
                    
                    # Let's be precise.
                    # Segment S = path[i+1 ... j].
                    # Before: path[i] -- path[i+1] ... path[j] -- path[j+1]
                    # After:  path[i] -- path[j] ... path[i+1] -- path[j+1]
                    
                    # Condition 1: path[i] must be connected to path[j]
                    if path[j] not in adj_filtered[path[i]]:
                        continue
                        
                    # Condition 2: 
                    # If j < L-1, path[i+1] must be connected to path[j+1]
                    if j < L-1:
                        if path[j+1] not in adj_filtered[path[i+1]]:
                            continue
                    else:
                        # If j is the last node, no new edge is formed at the right end.
                        # We just need to ensure we don't create a cycle or break the path.
                        # The path ends at path[i+1].
                        # This is a valid move if the internal connections hold (they do, due to reversal).
                        pass
                        
                    # Perform the reversal
                    path[i+1:j+1] = reversed(path[i+1:j+1])
                    improved = True
                    break
                if improved:
                    break
        return path

    # Main search loop
    # We will try multiple restarts with different initial paths
    # Initial path generation: Greedy or Random
    
    # Strategy:
    # 1. Generate a random valid path (start from a random node in largest comp, greedy extend).
    # 2. Apply 2-opt to rearrange.
    # 3. Extend.
    # 4. Repeat.
    
    rng = random.Random(42)
    nodes_list = list(largest_comp_nodes)
    
    # If the component is small, just do a simple DFS or exhaustive check if very small
    if largest_comp_size <= 10:
        # Simple DFS for small components
        def dfs_small(node, path, visited):
            nonlocal best_path, best_len
            if len(path) > best_len:
                best_path = list(path)
                best_len = len(path)
            for nb in adj_filtered[node]:
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs_small(nb, path, visited)
                    path.pop()
                    visited.remove(nb)
        
        for start in nodes_list:
            dfs_small(start, [start], {start})
            if time.time() > deadline:
                break
        return best_path

    # For larger components, use the hybrid metaheuristic
    for restart in range(20): # Max 20 restarts, but time limited
        if time.time() > deadline:
            break
            
        # Generate initial path
        # Start with a random node
        start = rng.choice(nodes_list)
        current_path = [start]
        visited = {start}
        
        # Greedy extension: pick neighbor with fewest unvisited neighbors (Warnsdorff-like)
        while True:
            last = current_path[-1]
            candidates = [nb for nb in adj_filtered[last] if nb not in visited]
            if not candidates:
                break
            # Sort by number of unvisited neighbors
            candidates.sort(key=lambda x: len([y for y in adj_filtered[x] if y not in visited]))
            next_node = candidates[0]
            current_path.append(next_node)
            visited.add(next_node)
            
        # Update best
        if len(current_path) > best_len:
            best_path = list(current_path)
            best_len = len(current_path)
            
        # Local Search Phase
        # Time budget for local search per restart
        ls_deadline = min(deadline, time.time() + 2.0) # 2 seconds per restart
        
        # Iterate 2-opt and extension
        for _ in range(50): # Max 50 iterations of local search
            if time.time() > ls_deadline:
                break
                
            # 2-opt
            current_path = two_opt(current_path, ls_deadline)
            
            # Extension
            current_path = extend_path(current_path)
            
            # Update best
            if len(current_path) > best_len:
                best_path = list(current_path)
                best_len = len(current_path)
                
            # If we reached the size of the component, we can stop
            if best_len == largest_comp_size:
                return best_path

    return best_path
