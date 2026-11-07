import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = P['n']
    # Normalize edges to (min, max) tuples for a set
    raw_edges = P['edges']
    edge_set = set()
    adj = {i: [] for i in range(n)}
    
    for u, v in raw_edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
    
    # We want to find 3 edge-disjoint Hamiltonian cycles
    cycles_found = []
    current_edge_set = set(edge_set)
    
    def find_hamiltonian_cycle(available_edges, start_node):
        """
        Finds a Hamiltonian cycle using DFS with Warnsdorff's heuristic.
        available_edges: set of (u, v) tuples with u < v
        start_node: int
        Returns: list of nodes in cycle, or None if not found
        """
        # Build adjacency for available edges only
        avail_adj = {i: [] for i in range(n)}
        for u, v in available_edges:
            avail_adj[u].append(v)
            avail_adj[v].append(u)
            
        # Precompute degrees for Warnsdorff (dynamic during search, but initial helps)
        # Actually, Warnsdorff is dynamic: prefer neighbors with fewer unused edges.
        
        path = [start_node]
        used_edges = set()
        
        def dfs(node):
            if time.time() > deadline:
                return False
            
            if len(path) == n:
                # Check if we can close the cycle
                last_node = path[-1]
                edge_key = (min(last_node, start_node), max(last_node, start_node))
                if edge_key in available_edges and edge_key not in used_edges:
                    return True
                return False
            
            next_node = path[-1]
            
            # Get neighbors
            neighbors = avail_adj[next_node]
            
            # Filter valid neighbors
            valid_neighbors = []
            for nb in neighbors:
                edge_key = (min(next_node, nb), max(next_node, nb))
                if edge_key in available_edges and edge_key not in used_edges:
                    if nb == start_node and len(path) < n:
                        continue # Cannot close early
                    valid_neighbors.append(nb)
            
            if not valid_neighbors:
                return False
            
            # Warnsdorff's Rule: Sort by number of available edges from the neighbor
            # Fewer available edges = more constrained = visit earlier
            def heuristic(nb):
                count = 0
                for nn in avail_adj[nb]:
                    ek = (min(nb, nn), max(nb, nn))
                    if ek in available_edges and ek not in used_edges:
                        if nn != start_node or len(path) == n - 1: # Allow closing edge count
                            count += 1
                return count
            
            # Add slight randomness to break ties or perturb order
            scored_neighbors = [(heuristic(nb), random.random(), nb) for nb in valid_neighbors]
            scored_neighbors.sort(key=lambda x: (x[0], x[1]))
            
            for _, _, nb in scored_neighbors:
                edge_key = (min(next_node, nb), max(next_node, nb))
                used_edges.add(edge_key)
                path.append(nb)
                
                if dfs(nb):
                    return True
                    
                path.pop()
                used_edges.remove(edge_key)
                
            return False
            
        if dfs(start_node):
            return path
        return None

    # Try to find 3 cycles
    for _ in range(3):
        if time.time() > deadline:
            break
            
        found = False
        # Retry a few times with different random seeds if the first attempt fails
        # The randomness comes from the shuffle in the heuristic sorting
        for attempt in range(100):
            cycle = find_hamiltonian_cycle(current_edge_set, 0)
            if cycle is not None:
                # Validate cycle
                is_valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    ek = (min(u, v), max(u, v))
                    if ek not in current_edge_set:
                        is_valid = False
                        break
                if is_valid and len(set(cycle)) == n:
                    cycles_found.append(cycle)
                    # Remove edges from current_edge_set
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        ek = (min(u, v), max(u, v))
                        current_edge_set.remove(ek)
                    found = True
                    break
            # If we run out of time, stop
            if time.time() > deadline:
                break
                
        if not found:
            break
            
    # Return the cycles found
    # The problem asks for a list of cycles.
    # If no cycles found, return empty? "nonempty list" is required.
    # The seed returns 1 cycle. We should at least return 1 if we can't find more.
    # But our search should find at least 1 easily.
    if not cycles_found:
        # Fallback: try to find just 1 with a simpler greedy or just return a known valid one if possible?
        # Given the graph, a Hamiltonian cycle almost certainly exists.
        # Let's try one more time without the deadline pressure for the first cycle if we have time
        if time.time() < deadline - 5:
             cycle = find_hamiltonian_cycle(edge_set, 0)
             if cycle:
                 cycles_found.append(cycle)
                 
    # Ensure types are correct
    # Vertices must be int
    # List must be list
    final_result = []
    for c in cycles_found:
        final_result.append([int(v) for v in c])
        
    if not final_result:
        # Absolute fallback, though highly unlikely
        # Just return a dummy valid cycle if we can construct one? 
        # The graph is connected.
        # Let's just return [] which is invalid per contract ("nonempty list").
        # We must ensure we return something.
        # If we truly fail, we might return a single node? No, "at least 3 distinct vertices".
        # If we fail to find any, we are in trouble.
        # But the seed program returns a valid cycle, so the graph has one.
        # Our algorithm is robust.
        pass
        
    return final_result
# EVOLVE-BLOCK-END
