"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = P['n']
    all_edges = P['edges']
    
    # Build adjacency list
    adj = defaultdict(set)
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def edge_key(u, v):
        if u > v:
            u, v = v, u
        return (u, v)
        
    # Pre-compute edge keys for all edges
    edge_keys = set()
    for u, v in all_edges:
        edge_keys.add(edge_key(u, v))
        
    def find_hamiltonian_cycle(available_edges_set, start, rng):
        """
        Find a Hamiltonian cycle in the graph defined by available_edges_set.
        Uses backtracking with Warnsdorff heuristic.
        """
        if len(available_edges_set) < n:
            return None
            
        # Check if start has at least 2 available neighbors
        start_neighbors = [v for v in adj[start] if edge_key(start, v) in available_edges_set]
        if len(start_neighbors) < 2:
            return None
            
        path = [start]
        visited = {start}
        used_edges = set()
        
        # For fast lookup of available edges from current node
        # We can just check available_edges_set
        
        def backtrack():
            if len(path) == n:
                # Check if we can close the cycle
                if edge_key(path[-1], start) in available_edges_set and edge_key(path[-1], start) not in used_edges:
                    return True
                return False
                
            current = path[-1]
            candidates = []
            
            for v in adj[current]:
                ek = edge_key(current, v)
                if v not in visited and ek in available_edges_set and ek not in used_edges:
                    # Heuristic: count unvisited neighbors of v
                    # This helps avoid isolating vertices
                    unvisited_count = 0
                    for w in adj[v]:
                        if w not in visited and edge_key(v, w) in available_edges_set and edge_key(v, w) not in used_edges:
                            unvisited_count += 1
                    # If unvisited_count is 0 and len(path) < n-1, this move isolates v (unless it's the last step, but we check len(path) < n)
                    # Actually, if unvisited_count == 0 and v is not the last vertex to be visited, it's a dead end.
                    # But we only add to candidates if it's a valid move.
                    candidates.append((unvisited_count, v))
                    
            if not candidates:
                return False
                
            # Sort by fewest unvisited neighbors (Warnsdorff)
            candidates.sort(key=lambda x: x[0])
            
            for _, v in candidates:
                ek = edge_key(current, v)
                path.append(v)
                visited.add(v)
                used_edges.add(ek)
                
                if backtrack():
                    return True
                
                path.pop()
                visited.remove(v)
                used_edges.discard(ek)
                
            return False
            
        if backtrack():
            return list(path)
        return None

    def find_cycle_with_restarts(available_edges_set, rng):
        """
        Try to find a Hamiltonian cycle by trying different start vertices and edge orderings.
        """
        vertices = list(range(n))
        
        # Sort vertices by degree in available graph (descending)
        def degree_in_avail(v):
            return sum(1 for u in adj[v] if edge_key(v, u) in available_edges_set)
            
        vertices.sort(key=lambda v: degree_in_avail(v), reverse=True)
        
        # Try each vertex as a start
        # We can also shuffle the order of tries
        starts = list(vertices)
        rng.shuffle(starts)
        
        for start in starts:
            cycle = find_hamiltonian_cycle(available_edges_set, start, rng)
            if cycle:
                return cycle
        return None

    # Main logic: Iteratively find edge-disjoint Hamiltonian cycles
    best_cycles = []
    available_edges_set = set(edge_keys)
    
    max_cycles = 3 # Theoretical upper bound for 6-regular graph
    
    for i in range(max_cycles):
        if time.time() > deadline:
            break
            
        if not available_edges_set:
            break
            
        found_cycle = None
        # Try multiple random restarts for the first cycle, fewer for subsequent ones if time is tight
        # But since n=21 is small, backtracking is fast.
        
        # To ensure we find a valid decomposition, we might need to be careful with which cycle we pick first.
        # However, for 3-edge-disjoint Hamiltonian cycles in a 6-regular graph, any valid first cycle should leave a 4-regular graph.
        # A 4-regular graph on 21 vertices might not have a Hamiltonian cycle? 
        # Wait, 21 vertices is odd. A 4-regular graph on odd vertices is possible.
        # Does it always have a Hamiltonian cycle? Not necessarily.
        # So the order of extraction matters.
        
        # Strategy: Try to find a cycle. If we get stuck later, we might need to backtrack the choice of the first cycle.
        # Given the time limit, we will try a few different "first" cycles.
        
        solutions = []
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Shuffle edges to bias the search? No, the search is on the graph structure.
            # We can randomize the start vertex selection.
            
            cycle = find_cycle_with_restarts(available_edges_set, random.Random(1234 + attempt))
            if cycle:
                # Verify cycle
                valid = True
                for j in range(len(cycle)):
                    u = cycle[j]
                    v = cycle[(j + 1) % len(cycle)]
                    if edge_key(u, v) not in available_edges_set:
                        valid = False
                        break
                if valid and len(set(cycle)) == n:
                    # Use this cycle and try to complete the rest
                    temp_cycles = [cycle]
                    temp_avail = set(available_edges_set)
                    
                    # Remove edges of cycle
                    for j in range(len(cycle)):
                        u = cycle[j]
                        v = cycle[(j + 1) % len(cycle)]
                        temp_avail.discard(edge_key(u, v))
                        
                    # Try to find remaining cycles
                    success = True
                    for k in range(1, max_cycles):
                        if time.time() > deadline:
                            success = False
                            break
                        if not temp_avail:
                            break
                            
                        next_cycle = find_cycle_with_restarts(temp_avail, random.Random(5678 + attempt + k))
                        if next_cycle:
                            # Verify
                            nv = True
                            for m in range(len(next_cycle)):
                                u = next_cycle[m]
                                v = next_cycle[(m + 1) % len(next_cycle)]
                                if edge_key(u, v) not in temp_avail:
                                    nv = False
                                    break
                            if nv and len(set(next_cycle)) == n:
                                temp_cycles.append(next_cycle)
                                for m in range(len(next_cycle)):
                                    u = next_cycle[m]
                                    v = next_cycle[(m + 1) % len(next_cycle)]
                                    temp_avail.discard(edge_key(u, v))
                            else:
                                success = False
                                break
                        else:
                            success = False
                            break
                            
                    if success and len(temp_cycles) > len(best_cycles):
                        best_cycles = temp_cycles
                        if len(best_cycles) == max_cycles:
                            return [[int(v) for v in c] for c in best_cycles]
        
        # If we didn't find 3 in the first pass, update best if we found something better
        if len(best_cycles) == max_cycles:
             return [[int(v) for v in c] for c in best_cycles]

    # If we still don't have 3, return what we have (hopefully 2 or 3)
    # The problem says "Maximize number of cycles".
    # If best_cycles is empty, we should try harder or return a single cycle if possible.
    # But find_cycle_with_restarts is quite robust.
    
    if not best_cycles:
        # Fallback: try to find at least 1
        cycle = find_cycle_with_restarts(available_edges_set, random.Random(999))
        if cycle:
            best_cycles = [cycle]
            
    return [[int(v) for v in c] for c in best_cycles]

# EVOLVE-BLOCK-END
