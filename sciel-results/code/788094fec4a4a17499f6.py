"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to lists for faster iteration
    adj_list = [list(s) for s in adj]
    
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
                for nb in adj_list[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
    
    # Sort by size, largest first
    components.sort(key=len, reverse=True)
    largest_comp = components[0]
    largest_set = set(largest_comp)
    
    # Helper: DFS to find a long simple path starting from a given node
    # Uses backtracking with a depth limit to avoid excessive search
    def dfs_longest_path(start, time_limit):
        """Find a long simple path starting from 'start' using DFS with backtracking."""
        best_path = [start]
        best_len = 1
        
        # Use iterative DFS with explicit stack to avoid recursion depth issues
        # Stack entries: (node, path_so_far, visited_set, next_neighbor_index)
        # For memory efficiency, we'll use a different approach
        
        # Better approach: recursive DFS with manual stack simulation
        # We'll use a list-based backtracking approach
        
        path = [start]
        visited_set = {start}
        
        # For each step, try to extend the path
        # If stuck, backtrack
        
        def try_extend(node, path, visited_set, depth, max_depth, time_limit):
            nonlocal best_path, best_len
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            if depth >= max_depth:
                return
            
            if time.time() > time_limit:
                return
            
            # Try neighbors in a randomized order for diversity
            neighbors = [nb for nb in adj_list[node] if nb not in visited_set]
            if not neighbors:
                return
            
            # Sort by degree (prefer lower degree neighbors first - Warnsdorff's heuristic)
            neighbors.sort(key=lambda x: len([y for y in adj_list[x] if y not in visited_set]))
            
            for nb in neighbors:
                if time.time() > time_limit:
                    return
                path.append(nb)
                visited_set.add(nb)
                try_extend(nb, path, visited_set, depth + 1, max_depth, time_limit)
                path.pop()
                visited_set.discard(nb)
        
        # Use a reasonable max depth
        max_depth = len(largest_comp)
        try_extend(start, path, visited_set, 0, max_depth, time_limit)
        
        return best_path
    
    # Helper: Greedy path with backtracking (limited)
    def greedy_with_backtrack(start, time_limit):
        """Greedy path construction with limited backtracking."""
        path = [start]
        visited_set = {start}
        
        while time.time() < time_limit:
            current = path[-1]
            # Find unvisited neighbors
            unvisited_neighbors = [nb for nb in adj_list[current] if nb not in visited_set]
            
            if not unvisited_neighbors:
                # Try backtracking
                if len(path) > 1:
                    # Pop last node and try to find an alternative
                    last = path.pop()
                    visited_set.discard(last)
                    # Mark that we've tried this node, so we don't get stuck
                    # For simplicity, just break if we can't extend
                    # Actually, let's try a different approach
                    break
                else:
                    break
            
            # Choose neighbor with fewest unvisited neighbors (Warnsdorff)
            unvisited_neighbors.sort(key=lambda x: len([y for y in adj_list[x] if y not in visited_set]))
            
            # Try the best neighbor
            next_node = unvisited_neighbors[0]
            path.append(next_node)
            visited_set.add(next_node)
        
        return path
    
    # Helper: Randomized DFS for finding long paths
    def randomized_dfs(start, seed, time_limit):
        """Randomized DFS to find a long path."""
        rng = random.Random(seed)
        path = [start]
        visited_set = {start}
        best_path = path[:]
        
        # Iterative DFS with backtracking
        stack = [(start, 0)]  # (node, next_neighbor_index)
        
        while stack and time.time() < time_limit:
            node, idx = stack[-1]
            
            # Get unvisited neighbors
            unvisited = [nb for nb in adj_list[node] if nb not in visited_set]
            
            if not unvisited:
                # Backtrack
                stack.pop()
                if stack:
                    prev_node, prev_idx = stack[-1]
                    # This node is done, remove from path
                    path.pop()
                    visited_set.discard(node)
                    # Update the parent's next index
                    # Actually, we need to track which neighbor we came from
                    # Let's use a different stack structure
                    break
                else:
                    break
            
            # Pick a random unvisited neighbor
            # But prefer lower degree for better paths
            unvisited.sort(key=lambda x: len([y for y in adj_list[x] if y not in visited_set]))
            
            # With some probability, pick randomly for diversity
            if rng.random() < 0.3:
                next_node = rng.choice(unvisited)
            else:
                next_node = unvisited[0]
            
            path.append(next_node)
            visited_set.add(next_node)
            
            if len(path) > len(best_path):
                best_path = path[:]
            
            stack.append((next_node, 0))
        
        return best_path
    
    # Strategy: Try multiple approaches and keep the best
    deadline = time.time() + 55  # 55 second deadline with safety margin
    best_overall = []
    
    # First, try a simple greedy from each node in the largest component
    for start in largest_comp:
        if time.time() > deadline:
            break
        path = greedy_with_backtrack(start, deadline)
        if len(path) > len(best_overall):
            best_overall = path
    
    # Now try DFS-based approaches from promising starting points
    # Start from nodes with high degree in the largest component
    degree_in_comp = {v: len([nb for nb in adj_list[v] if nb in largest_set]) for v in largest_comp}
    sorted_starts = sorted(largest_comp, key=lambda x: degree_in_comp[x], reverse=True)
    
    # Try DFS from top-degree nodes
    for start in sorted_starts[:20]:
        if time.time() > deadline:
            break
        path = dfs_longest_path(start, deadline)
        if len(path) > len(best_overall):
            best_overall = path
    
    # Try randomized DFS with multiple seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        start = sorted_starts[seed % len(sorted_starts)]
        path = randomized_dfs(start, seed, deadline)
        if len(path) > len(best_overall):
            best_overall = path
    
    # If we still haven't found a good path, try a more exhaustive search
    if len(best_overall) < 40 and time.time() < deadline - 5:
        # Try all pairs of nodes as potential path endpoints
        # This is expensive, so only do it if we have time
        for start in sorted_starts[:10]:
            if time.time() > deadline - 5:
                break
            # Do a deeper DFS
            path = dfs_longest_path(start, deadline - 2)
            if len(path) > len(best_overall):
                best_overall = path
    
    # Final fallback: if best_overall is empty or too short, use the greedy result
    if not best_overall:
        # Return a single vertex
        best_overall = [largest_comp[0]]
    
    return best_overall

# EVOLVE-BLOCK-END
