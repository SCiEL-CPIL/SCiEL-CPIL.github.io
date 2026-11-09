import time
import random
import sys

def construct():
    # Graph definition from seed
    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    cycles = []
    current_adj = {i: set(adj[i]) for i in range(n)}
    
    # Increase recursion limit for safety
    sys.setrecursionlimit(10000)
    
    def find_hamiltonian_cycle(adj_map, start_node=0):
        """
        Finds a Hamiltonian cycle in the graph defined by adj_map.
        Returns a list of vertices representing the cycle, or None if not found.
        """
        n_local = len(adj_map)
        path = [start_node]
        visited = {start_node}
        
        # Pre-check: all vertices must have degree >= 2
        for u in range(n_local):
            if len(adj_map[u]) < 2:
                return None
        
        # Precompute neighbor lists for speed
        adj_lists = {u: list(adj_map[u]) for u in range(n_local)}
        
        def backtrack():
            if time.time() > deadline:
                return False
                
            current = path[-1]
            
            if len(path) == n_local:
                # Check if we can close the cycle
                if start_node in adj_map[current]:
                    return True
                return False
            
            # Get unvisited neighbors
            unvisited_neighbors = [v for v in adj_lists[current] if v not in visited]
            
            if not unvisited_neighbors:
                return False
            
            # Heuristic: order neighbors by degree in the remaining graph
            # Lower degree first (Worst-First)
            # We approximate "remaining degree" by counting unvisited neighbors + (1 if start_node is unvisited? No, start is visited)
            # Actually, for a node v, its remaining connections are:
            # 1. Unvisited neighbors
            # 2. The node it came from (current) - but that's already visited
            # 3. start_node (if we are at the end)
            # A better heuristic: sort by the number of unvisited neighbors of v.
            # If v has few unvisited neighbors, it's more constrained.
            
            def degree_key(v):
                # Count unvisited neighbors of v
                cnt = 0
                for u in adj_lists[v]:
                    if u not in visited:
                        cnt += 1
                return cnt
            
            unvisited_neighbors.sort(key=degree_key)
            
            for next_node in unvisited_neighbors:
                # Pruning:
                # 1. If next_node is the last unvisited node, it must connect to start.
                if len(visited) + 1 == n_local:
                    if start_node not in adj_map[next_node]:
                        continue
                
                # 2. Check if moving to next_node isolates any unvisited node.
                # Specifically, if any unvisited node (other than next_node) has only 1 unvisited neighbor (current? no, current is visited) 
                # Wait, if we visit next_node, the "unvisited" set shrinks.
                # If there is an unvisited node w (w != next_node) that has only 1 neighbor in (unvisited U {next_node}), 
                # and that neighbor is next_node, then after visiting next_node, w will have 0 unvisited neighbors (and can't go to start unless it's the last step).
                # This is a strong pruning rule.
                
                # Let's implement a lighter version:
                # If next_node has only 1 unvisited neighbor (other than current), and that neighbor is not start (unless last step), 
                # we might be forced? No, we are CHOSING next_node.
                # The danger is: by visiting next_node, we might leave an unvisited node w with degree < 2 in the remaining graph.
                
                # Let's check: for all unvisited w != next_node:
                # deg_remaining(w) = count of unvisited neighbors of w (excluding next_node? No, next_node becomes visited)
                # So deg_remaining(w) = count of neighbors of w in (unvisited - {next_node})
                # Plus, if w can connect to start? Only if w is the last node.
                # Plus, if w can connect to current? No, current is visited.
                
                # Actually, the standard pruning is:
                # If we go to next_node, does any unvisited node w (w != next_node) have < 2 connections to (unvisited - {next_node}) U {start_node}?
                # Wait, start_node is already visited. The path ends at start_node.
                # So w must connect to at least one unvisited node (to continue path) and eventually to start_node (to close).
                # If w has 0 unvisited neighbors (after removing next_node from unvisited set), it can only be visited if it's the last node AND connects to start.
                # If w has 1 unvisited neighbor, it must be that neighbor (forced move).
                
                # This check is O(N^2). With N=21, it's fast.
                
                can_proceed = True
                for w in adj_lists[next_node]: # Only check neighbors of next_node? No, check all unvisited.
                    pass # Too slow to check all. Let's stick to simpler heuristics or just the last-step check.
                
                # Let's add a check for "dead ends" for next_node itself?
                # If next_node has only 1 unvisited neighbor (other than current), and we are not at the last step,
                # then we MUST go to that neighbor next. But we are just choosing next_node.
                # The constraint is on the FUTURE.
                
                # Let's just use the degree sorting and the last-step check. It's robust enough for N=21.
                
                visited.add(next_node)
                path.append(next_node)
                
                if backtrack():
                    return True
                
                path.pop()
                visited.remove(next_node)
                
            return False
        
        if backtrack():
            return path
        return None

    # Try to find 3 cycles
    # We might need to restart if we get stuck with < 3
    max_cycles = 3
    
    # Since the graph is symmetric, a specific order might work.
    # Let's just run the search.
    
    for i in range(max_cycles):
        if time.time() > deadline:
            break
        cycle = find_hamiltonian_cycle(current_adj)
        if cycle is None:
            break
        
        cycles.append(cycle)
        
        # Remove edges of this cycle from current_adj
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            
        # If we removed edges, check if graph is still 2-connected enough
        # If any vertex has degree < 2, we can't form another cycle
        valid = True
        for u in range(n):
            if len(current_adj[u]) < 2:
                valid = False
                break
        if not valid:
            break
            
    # If we found fewer than 3, try randomizing the start node or edge order?
    # The prompt says "Maximize number of cycles".
    # If we only found 2, maybe a different first cycle allows 3?
    # Given the time limit and the fact that 3 is the max, and previous runs found 3,
    # a simple greedy approach might suffice. 
    # However, to be robust, we can try a few random restarts if we don't get 3.
    
    if len(cycles) < 3 and time.time() < deadline - 5.0:
        # Try a few random restarts
        for attempt in range(10):
            if time.time() > deadline - 5.0:
                break
            temp_adj = {i: set(adj[i]) for i in range(n)}
            temp_cycles = []
            success = True
            for _ in range(3):
                # Random start node
                start_node = random.randint(0, n-1)
                cycle = find_hamiltonian_cycle(temp_adj, start_node)
                if cycle is None:
                    success = False
                    break
                temp_cycles.append(cycle)
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    temp_adj[u].remove(v)
                    temp_adj[v].remove(u)
            
            if success and len(temp_cycles) > len(cycles):
                cycles = temp_cycles
                if len(cycles) == 3:
                    break

    # Return the best found
    # Ensure types are correct: list of lists of ints
    return [list(c) for c in cycles]
