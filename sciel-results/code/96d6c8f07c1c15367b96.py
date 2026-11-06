"""Rewritten program: Backtracking search with Warnsdorff heuristic for longest path."""
import time
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find largest connected component
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
    
    # If the largest component is small, we might just return it
    # But we need a path. A tree is not necessarily a path.
    # We will run the search on the whole graph but restrict to nodes in the largest component
    # to avoid wasting time on small components.
    
    # Actually, the path must be contiguous. If we start in the largest component, we stay there.
    # We should identify the set of nodes in the largest component.
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
                
    # Filter adjacency to only include nodes in the largest component?
    # No, just use the set to check validity.
    
    start_time = time.time()
    deadline = start_time + 55.0
    best_path = []
    
    # Warnsdorff's rule: visit the neighbor with the fewest unvisited neighbors first
    # This is a heuristic for finding Hamiltonian paths.
    
    def dfs(node, path, visited):
        nonlocal best_path
        nonlocal deadline
        nonlocal start_time
        
        if time.time() > deadline:
            return False # Time up
            
        # Update best if current path is longer
        if len(path) > len(best_path):
            best_path = list(path)
            
        # Check if we have visited all nodes in the largest component
        if len(path) == len(largest_comp_nodes):
            return True # Found a path covering the whole component
            
        # Get unvisited neighbors
        unvisited_neighbors = [nb for nb in adj[node] if nb not in visited]
        
        if not unvisited_neighbors:
            return False # Dead end
            
        # Sort neighbors by Warnsdorff's rule: 
        # Priority: fewer unvisited neighbors first.
        # This helps avoid isolating vertices.
        def heuristic(nb):
            # Count unvisited neighbors of nb
            count = 0
            for n_nb in adj[nb]:
                if n_nb not in visited:
                    count += 1
            return (count, nb)
            
        unvisited_neighbors.sort(key=heuristic)
        
        for nb in unvisited_neighbors:
            if time.time() > deadline:
                return False
            visited.add(nb)
            path.append(nb)
            
            # Recursive call
            # We don't need to return True immediately if we found a full path, 
            # but we can stop searching if we found a path of length == len(largest_comp_nodes)
            # Actually, if we find a path of length == len(largest_comp_nodes), it's the best possible.
            # Let's check if we can terminate early.
            result = dfs(nb, path, visited)
            
            if result:
                return True
                
            path.pop()
            visited.remove(nb)
            
        return False

    # Run DFS from the start node of the largest component
    # It might be beneficial to try multiple start nodes if the first one fails to find a long path quickly.
    # But Warnsdorff is quite strong. Let's try one first.
    # If it times out, we return the best path found so far.
    
    # To improve success rate, we can try a few different start nodes if the first one doesn't yield a very long path quickly.
    # However, the graph is static. The "largest component" is the main target.
    # Let's try starting from the node with the lowest degree in the largest component? 
    # Or just the start_node identified above.
    
    # Let's try a few start nodes from the largest component.
    # We will run DFS for a fraction of the time for each, or just one long run.
    # One long run with backtracking is usually better than many short runs.
    
    # Let's pick a few candidate start nodes: 
    # 1. The one found by BFS (start_node)
    # 2. The one with min degree in the component
    # 3. The one with max degree in the component
    
    candidates = set()
    candidates.add(start_node)
    min_deg_node = min(largest_comp_nodes, key=lambda x: len(adj[x]))
    max_deg_node = max(largest_comp_nodes, key=lambda x: len(adj[x]))
    candidates.add(min_deg_node)
    candidates.add(max_deg_node)
    
    # Shuffle candidates to add variety? No, deterministic is better for reproducibility unless random is allowed.
    # The prompt says "random.Random(748)" is used for generation, but our code can be deterministic.
    # Let's just iterate through them.
    
    for start in candidates:
        if time.time() > deadline:
            break
        visited = {start}
        path = [start]
        # We don't want to restart from scratch if we already have a very good path.
        # But DFS backtracks, so it explores the tree. 
        # If we start from a different node, we explore a different tree.
        # If the first DFS finds a path of length L, and we start a new DFS, we might find L or more.
        # We should only start a new DFS if the previous one didn't find a path covering the whole component.
        if len(best_path) == len(largest_comp_nodes):
            break
            
        dfs(start, path, visited)
        
        # If we found a full path, we are done
        if len(best_path) == len(largest_comp_nodes):
            break

    return best_path
