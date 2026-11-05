"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Build adjacency
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
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
    
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    best_path = []
    
    # For each component, try to find longest path
    for comp in components:
        if time.time() > deadline:
            break
        if len(comp) <= len(best_path):
            continue
        
        comp_set = set(comp)
        comp_nodes = sorted(comp)
        
        # For small components, we can be more thorough
        # Use DFS with branch and bound
        best_len = len(best_path)
        
        # Try each vertex as start
        for start in comp_nodes:
            if time.time() > deadline:
                break
            
            # DFS with backtracking
            path = [start]
            on_path = [False] * n
            on_path[start] = True
            
            def dfs(current):
                nonlocal best_len, best_path
                if time.time() > deadline:
                    return
                
                # Update best if needed
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                
                # Pruning: can we possibly beat best_len?
                # Upper bound: len(path) + (number of unvisited vertices in comp)
                unvisited_count = len(comp) - len(path)
                if len(path) + unvisited_count <= best_len:
                    return
                
                # Try neighbors
                neighbors = [nb for nb in adj[current] if not on_path[nb]]
                # Sort by degree (heuristic: try low-degree first to avoid dead ends, 
                # or high-degree first to explore more)
                # Actually, for longest path, we want to explore branches that lead to more
                # vertices. Let's try higher degree first.
                neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                
                for nb in neighbors:
                    if time.time() > deadline:
                        return
                    on_path[nb] = True
                    path.append(nb)
                    dfs(nb)
                    path.pop()
                    on_path[nb] = False
            
            dfs(start)
    
    # If no path found (shouldn't happen), return a single vertex
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
