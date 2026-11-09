"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
    n = P['n']
    edges = [tuple(sorted(e)) for e in P['edges']]
    edge_set = set(edges)
    
    # Adjacency list for original graph
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    def find_hamiltonian_in(adj_map, deadline):
        """Find a Hamiltonian cycle in the graph defined by adj_map."""
        # Check degree: every vertex must have degree >= 2
        for i in range(n):
            if len(adj_map[i]) < 2:
                return None
        
        # Check connectivity
        visited = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for v in adj_map[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        if len(visited) < n:
            return None
        
        # Multiple randomized backtracking attempts
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            rng = random.Random(attempt * 7919 + 13)
            path = [0]
            visited = {0}
            
            def backtrack(u, deadline):
                if len(path) == n:
                    if 0 in adj_map[u]:
                        return True
                    return False
                
                # Get unvisited neighbors
                candidates = [v for v in adj_map[u] if v not in visited]
                if not candidates:
                    return False
                
                # Warnsdorff's rule: prefer vertices with fewer onward moves
                candidates.sort(key=lambda x: len(adj_map[x] - visited))
                
                # Randomize order based on attempt
                if attempt > 0:
                    rng.shuffle(candidates)
                
                for v in candidates:
                    path.append(v)
                    visited.add(v)
                    if backtrack(v, deadline):
                        return True
                    visited.remove(v)
                    path.pop()
                return False
            
            if backtrack(0, deadline):
                # Verify the cycle
                for i in range(n):
                    u = path[i]
                    v = path[(i + 1) % n]
                    if v not in adj_map[u]:
                        return None
                return path
        return None
    
    def get_cycle_edges(cycle):
        """Get set of edges in a cycle."""
        edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edges.add(tuple(sorted((u, v))))
        return edges
    
    def remove_cycle_edges(rem_edges, cycle):
        """Remove cycle edges from rem_edges set."""
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e in rem_edges:
                rem_edges.remove(e)
    
    # Greedy search for maximum number of edge-disjoint Hamiltonian cycles
    best_sol = []
    
    # Try multiple random orderings of greedy cycle selection
    for outer_attempt in range(20):
        if time.time() > deadline:
            break
        
        rng = random.Random(outer_attempt * 104729 + 42)
        rem_edges = set(edge_set)
        cycles = []
        
        while time.time() < deadline:
            # Build adjacency for remaining edges
            rem_adj = {i: set() for i in range(n)}
            for u, v in rem_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Find a Hamiltonian cycle in the remaining graph
            c = find_hamiltonian_in(rem_adj, deadline)
            if c is None:
                break
            
            cycles.append(c)
            
            # Remove edges of this cycle from remaining edges
            remove_cycle_edges(rem_edges, c)
            
            # Update best solution
            if len(cycles) > len(best_sol):
                best_sol = [list(c) for c in cycles]
            
            # If we've reached the theoretical maximum, stop
            if len(best_sol) == 3:
                return best_sol
    
    # Return best solution found, ensuring proper types
    result = []
    for c in best_sol:
        result.append([int(v) for v in c])
    
    if not result:
        # Fallback: find any Hamiltonian cycle in the original graph
        c = find_hamiltonian_in(adj, deadline)
        if c:
            result = [[int(v) for v in c]]
        else:
            # Last resort: return a trivial valid cycle
            # Find any 3-cycle
            for e1 in edges:
                for e2 in edges:
                    if e1[1] == e2[0] and e1 != e2:
                        if (e1[0], e2[1]) in edge_set or (e2[1], e1[0]) in edge_set:
                            result = [[int(e1[0]), int(e1[1]), int(e2[1])]]
                            break
                if result:
                    break
            if not result:
                result = [[0, 1, 2]]  # Should not happen
    
    return result
# EVOLVE-BLOCK-END
