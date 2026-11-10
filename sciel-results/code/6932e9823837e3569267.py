"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85  # Leave 5s safety margin
    
    # Build edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    # Adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # First, find two known Hamiltonian cycles
    # Cycle 1: 0-1-2-...-20-0
    c1 = list(range(n))
    edges_c1 = set()
    for i in range(n):
        u, v = c1[i], c1[(i+1)%n]
        if u > v: u, v = v, u
        edges_c1.add((u, v))
    
    # Cycle 2: 0-2-4-...-20-1-3-...-19-0
    c2 = []
    for i in range(0, n, 2):
        c2.append(i)
    for i in range(1, n, 2):
        c2.append(i)
    edges_c2 = set()
    for i in range(n):
        u, v = c2[i], c2[(i+1)%n]
        if u > v: u, v = v, u
        edges_c2.add((u, v))
    
    if edges_c1 & edges_c2:
        # Fallback: just return c1
        return [c1]
    
    # Remaining edges after removing c1 and c2
    remaining_edges = edge_set - edges_c1 - edges_c2
    
    # Build adjacency for remaining graph
    rem_adj = {i: set() for i in range(n)}
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    # Now use CSP approach to find a 3rd Hamiltonian cycle in the remaining graph
    # The remaining graph should be 1-regular if we found 2 perfect matchings... 
    # Wait, the original graph is 5-regular. After removing 2 Hamiltonian cycles (each uses 21 edges),
    # we have 5*21/2 - 2*21 = 52.5 - 42 = 10.5... that doesn't work.
    # Let me recalculate: 5-regular graph on 21 vertices has 5*21/2 = 52.5 edges? No, that's not integer.
    # Actually 5*21 = 105, divided by 2 = 52.5. That can't be right for a simple graph.
    # Let me count the edges in P:
    # Looking at the edge list... let me just count them.
    
    # Actually, let me just try to find a Hamiltonian cycle in the remaining graph using DFS with good heuristics
    
    # Better approach: use backtracking with constraint propagation
    # Each vertex must have exactly 2 incident edges in the cycle
    
    def find_hamiltonian_cycle(adj_graph, deadline):
        """Find a Hamiltonian cycle using DFS with degree-based pruning."""
        n_local = len(adj_graph)
        
        # For each vertex, track how many edges are "used" in the partial solution
        # and how many are still "available"
        
        # We'll use a different approach: build the cycle edge by edge
        # Start with vertex 0, and try to extend the path
        
        # Use memoization-like pruning: if a vertex has only one available neighbor 
        # that isn't the start/end, we must use that edge
        
        def dfs(path, used_edges, available):
            """
            path: current path from start
            used_edges: set of edges used so far
            available: dict vertex -> set of available neighbors (not yet used in cycle)
            """
            if time.time() > deadline:
                return None
            
            if len(path) == n_local:
                # Check if we can close the cycle
                last = path[-1]
                first = path[0]
                if first in available[last]:
                    return path[:]
                return None
            
            last = path[-1]
            
            # Get available neighbors
            neighbors = list(available[last])
            
            # Heuristic: prefer neighbors with fewer available options (most constrained first)
            neighbors.sort(key=lambda x: len(available[x]) if x != path[0] else 0)
            
            for v in neighbors:
                if v in path and v != path[0]:
                    continue
                if v == path[0] and len(path) < n_local - 1:
                    continue
                
                # Try using edge (last, v)
                new_used = used_edges | {(min(last,v), max(last,v))}
                
                # Update available: remove last-v edge from both
                new_available = {k: set(val) for k, val in available.items()}
                new_available[last].discard(v)
                new_available[v].discard(last)
                
                # Check if this creates a premature cycle (not the full Hamiltonian cycle)
                # If v is not the start and v is in path, we'd create a cycle - but we already checked
                # Also check if any vertex now has 0 available edges and isn't the start or end of path
                valid = True
                for node in path[1:-1]:  # Internal nodes of the path
                    if len(new_available[node]) < 1:
                        # This node needs at least one more edge to connect back
                        # Actually, internal nodes need exactly 2 total edges, they already have 2 (prev and next in path)
                        # So they should have 0 available edges - that's fine
                        pass
                # Check nodes not in path: they need at least 2 available edges
                for node in range(n_local):
                    if node not in path and node != v:
                        if len(new_available[node]) < 2:
                            valid = False
                            break
                    elif node == v and v != path[0]:
                        # v is now in the path, it has one edge used (from last)
                        # It needs one more edge
                        if len(new_available[v]) < 1:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Special case: if v is the start, we're closing the cycle
                if v == path[0]:
                    if len(path) == n_local:
                        return path[:]
                    else:
                        continue
                
                res = dfs(path + [v], new_used, new_available)
                if res is not None:
                    return res
            
            return None
        
        # Initialize available
        init_available = {i: set(adj_graph[i]) for i in range(n_local)}
        
        # Start with vertex 0
        start = 0
        init_path = [start]
        # Remove start from its own available (self-loops don't exist)
        # The start vertex will have 2 edges in the final cycle
        
        return dfs(init_path, set(), init_available)
    
    # Try to find the third Hamiltonian cycle
    c3 = find_hamiltonian_cycle(rem_adj, deadline)
    
    if c3 is None and time.time() < deadline:
        # Try randomized restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
            random.seed(attempt)
            # Shuffle the order of neighbors in rem_adj
            shuffled_adj = {i: list(rem_adj[i]) for i in range(n)}
            for i in shuffled_adj:
                random.shuffle(shuffled_adj[i])
            # Convert back to sets but with shuffled iteration order
            # Actually, let's just try a different starting vertex
            start_vertex = random.randint(0, n-1)
            
            # Modified DFS with different start
            def dfs_with_start(start_v, adj_graph, deadline):
                n_local = len(adj_graph)
                
                def dfs(path, used_edges, available):
                    if time.time() > deadline:
                        return None
                    
                    if len(path) == n_local:
                        last = path[-1]
                        first = path[0]
                        if first in available[last]:
                            return path[:]
                        return None
                    
                    last = path[-1]
                    neighbors = list(available[last])
                    neighbors.sort(key=lambda x: len(available[x]) if x != path[0] else 0)
                    
                    for v in neighbors:
                        if v in path and v != path[0]:
                            continue
                        if v == path[0] and len(path) < n_local - 1:
                            continue
                        
                        new_used = used_edges | {(min(last,v), max(last,v))}
                        new_available = {k: set(val) for k, val in available.items()}
                        new_available[last].discard(v)
                        new_available[v].discard(last)
                        
                        valid = True
                        for node in range(n_local):
                            if node not in path and node != v:
                                if len(new_available[node]) < 2:
                                    valid = False
                                    break
                            elif node == v and v != path[0]:
                                if len(new_available[v]) < 1:
                                    valid = False
                                    break
                        
                        if not valid:
                            continue
                        
                        if v == path[0]:
                            if len(path) == n_local:
                                return path[:]
                            else:
                                continue
                        
                        res = dfs(path + [v], new_used, new_available)
                        if res is not None:
                            return res
                    
                    return None
                
                init_available = {i: set(adj_graph[i]) for i in range(n_local)}
                return dfs([start_v], set(), init_available)
            
            c3 = dfs_with_start(start_vertex, rem_adj, deadline)
            if c3 is not None:
                break
    
    # Validate c3
    if c3 is not None:
        edges_c3 = set()
        valid = True
        for i in range(len(c3)):
            u, v = c3[i], c3[(i+1)%len(c3)]
            if u > v: u, v = v, u
            if (u, v) not in remaining_edges:
                valid = False
                break
            edges_c3.add((u, v))
        
        if valid and len(edges_c3) == len(c3) and len(c3) == n:
            return [c1, c2, c3]
        else:
            return [c1, c2]
    else:
        return [c1, c2]

# EVOLVE-BLOCK-END
