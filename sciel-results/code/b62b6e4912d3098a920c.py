"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    def find_hamiltonian_cycle(adj_map, start_node=0, deadline=None):
        """
        Finds a Hamiltonian cycle in the graph defined by adj_map.
        Uses DFS with backtracking and degree-based neighbor ordering.
        Returns a list of vertices representing the cycle, or None if not found.
        """
        if deadline is None:
            deadline = start_time + 80.0
        
        n_local = len(adj_map)
        
        # Pre-check: all vertices must have degree >= 2
        for u in range(n_local):
            if len(adj_map[u]) < 2:
                return None
        
        # Check connectivity of the remaining graph
        visited_conn = set()
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node not in visited_conn:
                visited_conn.add(node)
                for nb in adj_map[node]:
                    if nb not in visited_conn:
                        stack.append(nb)
        if len(visited_conn) < n_local:
            return None
        
        path = [start_node]
        visited = {start_node}
        
        # Precompute degrees for ordering
        def get_priority(v, visited_set, current):
            # Count unvisited neighbors of v
            deg = 0
            for w in adj_map[v]:
                if w not in visited_set and w != current:
                    deg += 1
            return deg
        
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
            unvisited_neighbors = [v for v in adj_map[current] if v not in visited]
            
            if not unvisited_neighbors:
                return False
            
            # Sort neighbors by their degree in the remaining unvisited subgraph
            # Lower degree first (most constrained vertices)
            unvisited_neighbors.sort(key=lambda v: get_priority(v, visited, current))
            
            for next_node in unvisited_neighbors:
                # Pruning: if this is the last vertex, it must connect to start
                if len(visited) + 1 == n_local:
                    if start_node not in adj_map[next_node]:
                        continue
                
                # Additional pruning: check if any unvisited vertex becomes isolated
                if len(visited) > n_local // 2:
                    stuck = False
                    for u in range(n_local):
                        if u in visited:
                            continue
                        # Count unvisited neighbors of u (excluding the one we're about to visit)
                        unvisited_count = sum(1 for w in adj_map[u] if w not in visited and w != next_node)
                        # u can also connect to start if it's the last vertex
                        if unvisited_count == 0:
                            # Check if u can connect to start (only valid if u is the last vertex)
                            if len(visited) + 1 == n_local - 1:
                                # u would be the last vertex, so it must connect to start
                                if start_node not in adj_map[u]:
                                    stuck = True
                                    break
                            else:
                                # u would be stuck in the middle
                                stuck = True
                                break
                    if stuck:
                        continue
                
                visited.add(next_node)
                path.append(next_node)
                
                if backtrack():
                    return True
                
                path.pop()
                visited.remove(next_node)
                
            return False
        
        if backtrack():
            return path[:]
        return None
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles (graph is 6-regular)
    max_cycles = 3
    best_cycles = []
    
    # First attempt: deterministic with start=0
    current_adj = {i: set(adj[i]) for i in range(n)}
    cycles = []
    
    for i in range(max_cycles):
        if time.time() > deadline:
            break
        
        cycle = find_hamiltonian_cycle(current_adj, start_node=0, deadline=deadline)
        if cycle is None:
            # Try other starting vertices
            found = False
            for start in range(1, n):
                if time.time() > deadline:
                    break
                cycle = find_hamiltonian_cycle(current_adj, start_node=start, deadline=deadline)
                if cycle is not None:
                    found = True
                    break
            if not found:
                break
        
        cycles.append(cycle)
        
        # Remove edges of this cycle from current_adj
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            current_adj[u].discard(v)
            current_adj[v].discard(u)
        
        # Check if graph is still viable
        valid = True
        for u in range(n):
            if len(current_adj[u]) < 2:
                valid = False
                break
        if not valid:
            break
    
    best_cycles = cycles[:]
    
    # If we didn't find 3, try random restarts
    if len(best_cycles) < max_cycles:
        random.seed(42)
        attempts = 0
        max_attempts = 100
        
        while time.time() < deadline - 5.0 and attempts < max_attempts:
            attempts += 1
            temp_adj = {i: set(adj[i]) for i in range(n)}
            temp_cycles = []
            success = True
            
            for _ in range(max_cycles):
                if time.time() > deadline - 5.0:
                    success = False
                    break
                
                # Try multiple start nodes
                cycle = None
                start_nodes = [random.randint(0, n - 1)]
                for s in random.sample(range(n), min(5, n)):
                    start_nodes.append(s)
                
                for start_node in start_nodes:
                    if time.time() > deadline - 5.0:
                        break
                    cycle = find_hamiltonian_cycle(temp_adj, start_node=start_node, deadline=deadline)
                    if cycle is not None:
                        break
                
                if cycle is None:
                    success = False
                    break
                
                temp_cycles.append(cycle)
                
                # Remove edges
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    temp_adj[u].discard(v)
                    temp_adj[v].discard(u)
                
                # Check viability
                valid = True
                for u in range(n):
                    if len(temp_adj[u]) < 2:
                        valid = False
                        break
                if not valid:
                    success = False
                    break
            
            if success and len(temp_cycles) > len(best_cycles):
                best_cycles = temp_cycles[:]
                if len(best_cycles) == max_cycles:
                    break
    
    # Fallback: ensure we return at least one valid cycle
    if not best_cycles:
        # Try one more time with a simple cycle
        best_cycles = [[i for i in range(n)]]
        # Verify it's valid
        valid = True
        for i in range(n):
            u = best_cycles[0][i]
            v = best_cycles[0][(i + 1) % n]
            if v not in adj[u]:
                valid = False
                break
        if not valid:
            # Find any valid cycle
            for start in range(n):
                cycle = find_hamiltonian_cycle({i: set(adj[i]) for i in range(n)}, start_node=start, deadline=deadline)
                if cycle is not None:
                    best_cycles = [cycle]
                    break
            if not best_cycles:
                best_cycles = [[0, 1, 2]]  # Minimal fallback
    
    # Verify edge-disjointness
    used_edges = set()
    valid_result = []
    for cycle in best_cycles:
        cycle_edges = set()
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            edge = (min(u, v), max(u, v))
            if edge in used_edges:
                # Edge conflict, stop adding
                break
            cycle_edges.add(edge)
        else:
            # No conflict, add this cycle
            used_edges.update(cycle_edges)
            valid_result.append(cycle)
    
    # Convert to list of lists of Python ints
    result = [list(c) for c in valid_result]
    
    return result
# EVOLVE-BLOCK-END
