import time
import random

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
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
            
    # Get the largest component
    if not components:
        return []
    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)
    
    # If the largest component is small, just return it (as a path? No, must be a path)
    # We need to find a path within this component.
    
    best_path = []
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    # Helper to check time
    def time_up():
        return time.time() > deadline

    # We will perform a DFS with backtracking and Warnsdorff heuristic.
    # To avoid recursion depth issues and allow for restarts, we use an iterative approach or careful recursion.
    # Given N=100, recursion is fine if we limit depth, but backtracking can be deep.
    # Let's use a stack for the DFS state.
    
    # State: (path, seen_set, next_index_to_try)
    # To make it efficient, we sort neighbors once per node visit? No, seen changes.
    
    # We'll try multiple start nodes from the largest component.
    # Shuffle the start nodes to diversify search.
    starts = list(largest_comp)
    random.seed(748) # Pinned seed for reproducibility if needed, but randomization helps search
    random.shuffle(starts)
    
    # We can also try to start from nodes with low degree first, as they are harder to place later.
    # Sort starts by degree
    starts.sort(key=lambda x: len(adj[x]))
    
    for start in starts:
        if time_up():
            break
            
        # Initialize search
        path = [start]
        seen = {start}
        # Stack for backtracking: stores (node, next_neighbor_index, sorted_neighbors)
        # We need to re-evaluate neighbors at each step because 'seen' changes.
        
        # A more efficient way: Recursive DFS with global best tracking.
        # But we need to limit the search space.
        
        def dfs(current_node, path, seen):
            nonlocal best_path
            if time_up():
                return
            
            if len(path) > len(best_path):
                best_path = path[:]
                
            # Get unvisited neighbors
            unvisited_neighbors = [v for v in adj[current_node] if v not in seen]
            if not unvisited_neighbors:
                return
            
            # Sort by Warnsdorff's rule: fewest unvisited neighbors first
            # Tie-breaker: vertex id for determinism, or random
            unvisited_neighbors.sort(key=lambda v: (len([u for u in adj[v] if u not in seen]), v))
            
            for next_node in unvisited_neighbors:
                if time_up():
                    return
                path.append(next_node)
                seen.add(next_node)
                dfs(next_node, path, seen)
                path.pop()
                seen.remove(next_node)
                
                # If we found a path that covers all nodes in the component, we can stop early
                if len(best_path) == len(largest_comp):
                    return

        # To prevent the DFS from getting stuck in a local optimum for too long,
        # we can limit the number of backtracking steps or time per start node.
        # However, the global time limit handles this.
        # To diversify, we can also add a small randomization to the sort key.
        
        dfs(start, path, seen)
        
        # If we found a very long path, we might want to stop if it's close to max possible
        if len(best_path) >= 0.9 * len(largest_comp):
            break
            
    return best_path
# EVOLVE-BLOCK-END
