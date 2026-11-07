"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    # Try to find k edge-disjoint Hamiltonian cycles
    # Use backtracking with edge assignment
    
    def find_hamiltonian_cycles(max_cycles=3):
        """Find up to max_cycles edge-disjoint Hamiltonian cycles."""
        used_edges = set()
        cycles = []
        
        def find_one_cycle(used_edges, cycles, remaining_deadline):
            """Find one Hamiltonian cycle using only unused edges."""
            if time.time() > remaining_deadline:
                return None
            
            # Use DFS to find a Hamiltonian cycle
            # Start from vertex 0, try to visit all vertices and return to 0
            n = P['n']
            visited = [False] * n
            path = [0]
            visited[0] = True
            
            def dfs(current, path, visited, remaining_deadline):
                if time.time() > remaining_deadline:
                    return None
                
                if len(path) == n:
                    # Check if we can close the cycle
                    edge = (min(0, current), max(0, current))
                    if edge not in used_edges:
                        return path + [0]
                    return None
                
                for neighbor in adj[current]:
                    if not visited[neighbor]:
                        edge = (min(current, neighbor), max(current, neighbor))
                        if edge not in used_edges:
                            visited[neighbor] = True
                            path.append(neighbor)
                            result = dfs(neighbor, path, visited, remaining_deadline)
                            if result is not None:
                                return result
                            path.pop()
                            visited[neighbor] = False
                
                return None
            
            result = dfs(0, path, visited, remaining_deadline)
            return result
        
        # Greedily find cycles
        for _ in range(max_cycles):
            if time.time() > deadline:
                break
            cycle = find_one_cycle(used_edges, cycles, deadline)
            if cycle is None:
                break
            # Mark edges as used
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i+1]
                edge = (min(u,v), max(u,v))
                used_edges.add(edge)
            cycles.append(cycle)
        
        return cycles
    
    # First try greedy approach
    cycles = find_hamiltonian_cycles(3)
    
    if len(cycles) >= 3:
        # Convert to proper format
        result = []
        for cycle in cycles:
            # Remove the closing 0 if present, keep as list of distinct vertices
            if cycle and cycle[0] == cycle[-1]:
                cycle = cycle[:-1]
            # Ensure all vertices are Python ints
            cycle = [int(v) for v in cycle]
            result.append(cycle)
        return result[:10]
    
    # If greedy didn't find 3, try backtracking with more careful search
    # This is a fallback - try to find 3 cycles with backtracking on edge assignments
    
    def find_three_cycles_backtrack():
        """Backtracking search for 3 edge-disjoint Hamiltonian cycles."""
        n = P['n']
        all_edges = list(edge_set)
        
        # State: for each edge, which cycle (0, 1, 2) or -1 if unused
        # We'll build cycles one at a time
        
        def find_cycle_with_constraints(used_edges, deadline):
            """Find a Hamiltonian cycle avoiding used_edges."""
            visited = [False] * n
            path = [0]
            visited[0] = True
            
            def dfs(current, path, visited, deadline):
                if time.time() > deadline:
                    return None
                
                if len(path) == n:
                    edge = (min(0, current), max(0, current))
                    if edge not in used_edges:
                        return list(path)
                    return None
                
                for neighbor in sorted(adj[current]):
                    if not visited[neighbor]:
                        edge = (min(current, neighbor), max(current, neighbor))
                        if edge not in used_edges:
                            visited[neighbor] = True
                            path.append(neighbor)
                            result = dfs(neighbor, path, visited, deadline)
                            if result is not None:
                                return result
                            path.pop()
                            visited[neighbor] = False
                
                return None
            
            return dfs(0, path, visited, deadline)
        
        used_edges = set()
        cycles = []
        
        for _ in range(3):
            if time.time() > deadline:
                break
            cycle = find_cycle_with_constraints(used_edges, deadline)
            if cycle is None:
                break
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i+1]
                edge = (min(u,v), max(u,v))
                used_edges.add(edge)
            # Also add closing edge
            u, v = cycle[-1], cycle[0]
            edge = (min(u,v), max(u,v))
            used_edges.add(edge)
            cycles.append(cycle)
        
        return cycles
    
    cycles = find_three_cycles_backtrack()
    
    if len(cycles) >= 1:
        result = []
        for cycle in cycles:
            cycle = [int(v) for v in cycle]
            result.append(cycle)
        return result[:10]
    
    # Fallback: return single cycle
    return [list(range(n))]
# EVOLVE-BLOCK-END
