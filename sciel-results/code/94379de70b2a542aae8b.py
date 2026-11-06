import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Convert to list of lists for faster iteration, maybe sort later
    adj_list = [list(s) for s in adj]
    
    best_path = []
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # Helper to check time
    def time_check():
        return time.time() < deadline

    # Randomized DFS with Heuristic Ordering
    # We will run this multiple times with different random seeds/orderings
    # The heuristic: visit neighbors with fewer unvisited neighbors first (Warnsdorff)
    
    def dfs(node, path, visited, rng):
        nonlocal best_path
        
        # Pruning: if current path + remaining unvisited <= best, stop
        # This is a weak bound but helps
        if len(path) + (n - len(visited)) <= len(best_path):
            return
        
        if not time_check():
            return
            
        # Update best
        if len(path) > len(best_path):
            best_path = path[:]
            
        # Get unvisited neighbors
        unvisited_neighbors = [u for u in adj_list[node] if u not in visited]
        
        if not unvisited_neighbors:
            return
            
        # Sort by number of unvisited neighbors of the neighbor (Warnsdorff)
        # We want to visit "dead ends" first to keep the path open for the rest
        # But we also want to explore promising branches.
        # Let's sort by (unvisited_degree, random_noise)
        
        def get_score(neighbor):
            # Number of unvisited neighbors of the neighbor
            deg = sum(1 for u in adj_list[neighbor] if u not in visited)
            # Add some randomness to break ties and explore different paths
            return (deg, rng.random())
            
        unvisited_neighbors.sort(key=get_score)
        
        for next_node in unvisited_neighbors:
            path.append(next_node)
            visited.add(next_node)
            
            dfs(next_node, path, visited, rng)
            
            path.pop()
            visited.remove(next_node)
            
            if not time_check():
                return

    # Run multiple randomized searches
    # We can vary the starting node or the random seed
    # Since the graph is small, we can afford many restarts
    
    # First, do a deterministic run to get a baseline quickly
    # Then do randomized runs
    
    # Deterministic run (Warnsdorff, no random tie break, or fixed tie break)
    # Actually, let's just use randomized runs with different seeds
    
    # To ensure we cover the graph well, we can start from different nodes
    # But DFS from one node in a connected component will cover that component.
    # We should identify connected components?
    # Or just start from all nodes? That's 100 starts.
    # With backtracking, 100 starts might be too slow if each is deep.
    # Let's try starting from a few random nodes, or all nodes but with strict time limits per start?
    # Better: One long run with many internal restarts? No, DFS is recursive.
    
    # Strategy:
    # 1. Find connected components.
    # 2. For the largest component, run randomized DFS.
    # 3. If time remains, run on other components or re-run on largest with different seeds.
    
    # Find connected components
    visited_comp = set()
    components = []
    for i in range(n):
        if i not in visited_comp:
            comp = []
            stack = [i]
            visited_comp.add(i)
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj_list[node]:
                    if nei not in visited_comp:
                        visited_comp.add(nei)
                        stack.append(nei)
            components.append(comp)
            
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    # We focus on the largest component primarily
    # But we can also just run DFS on the whole graph, it will just not traverse between components.
    
    # Let's run randomized DFS starting from different nodes in the largest component
    # To maximize coverage, we can pick a random start node each time.
    
    # We will run a loop of searches until time runs out
    search_count = 0
    while time_check():
        search_count += 1
        # Reset best_path for this specific search? 
        # No, we want the GLOBAL best.
        # But DFS is depth-first. It finds *a* path.
        # If we keep the global best, we might prune branches that could lead to a longer path 
        # if the current best is already very long.
        # However, `len(path) + (n - len(visited)) <= len(best_path)` is a valid prune.
        
        # Pick a random start node from the largest component
        if components:
            start_node = random.choice(components[0])
        else:
            start_node = 0
            
        path = [start_node]
        visited = {start_node}
        
        # Use a new RNG for each search to ensure diversity
        # Seed it with search_count or time
        rng = random.Random(search_count)
        
        dfs(start_node, path, visited, rng)
        
        # Optional: If we found a path that covers the whole component, we can stop
        if len(best_path) == len(components[0]):
            break
            
    return best_path
