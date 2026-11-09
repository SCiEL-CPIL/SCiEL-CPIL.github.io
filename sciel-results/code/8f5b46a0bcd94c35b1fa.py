"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency and edge sets
    adj = {i: set() for i in range(n)}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
        
    all_vertices = list(range(n))
    
    def get_cycle_edges(cycle):
        return [(min(cycle[i], cycle[(i+1)%len(cycle)]), max(cycle[i], cycle[(i+1)%len(cycle)])) for i in range(len(cycle))]

    def is_hamiltonian_cycle(cycle, forbidden_edges):
        if len(cycle) != n:
            return False
        if len(set(cycle)) != n:
            return False
        for e in get_cycle_edges(cycle):
            if e in forbidden_edges:
                return False
            if e not in edge_set:
                return False
        return True

    def find_ham_cycle(forbidden_edges, max_attempts=100):
        """Find a Hamiltonian cycle avoiding forbidden_edges using Warnsdorff's heuristic."""
        for _ in range(max_attempts):
            start = random.choice(all_vertices)
            path = [start]
            visited = {start}
            
            def backtrack(path, visited):
                if len(path) == n:
                    last = path[-1]
                    first = path[0]
                    e = (min(last, first), max(last, first))
                    if e in forbidden_edges or e not in edge_set:
                        return False
                    return True
                
                current = path[-1]
                # Candidates: neighbors not visited, edge not forbidden
                candidates = []
                for v in adj[current]:
                    if v not in visited:
                        e = (min(current, v), max(current, v))
                        if e not in forbidden_edges and e in edge_set:
                            # Heuristic: prefer vertices with fewer available exits
                            exits = 0
                            for w in adj[v]:
                                if w not in visited and w != current:
                                    ew = (min(v, w), max(v, w))
                                    if ew not in forbidden_edges and ew in edge_set:
                                        exits += 1
                            candidates.append((exits, v))
                
                if not candidates:
                    return False
                    
                # Sort by fewest exits (Warnsdorff)
                candidates.sort(key=lambda x: x[0])
                
                # Add some randomness to tie-breaking
                for _, v in candidates:
                    path.append(v)
                    visited.add(v)
                    if backtrack(path, visited):
                        return True
                    path.pop()
                    visited.remove(v)
                
                return False
            
            if backtrack(path, visited):
                return path
        return None

    def check_remainder_is_hamiltonian(used_edges):
        """Check if the remaining edges form a single Hamiltonian cycle."""
        remaining_edges = edge_set - used_edges
        if len(remaining_edges) != n:
            return None
        
        # Build adjacency of remaining graph
        rem_adj = {i: [] for i in range(n)}
        for u, v in remaining_edges:
            rem_adj[u].append(v)
            rem_adj[v].append(u)
            
        # Check if it's a single cycle of length n
        # Start from 0
        if len(rem_adj[0]) != 2:
            return None
            
        path = [0]
        visited = {0}
        current = 0
        for _ in range(n):
            nexts = [v for v in rem_adj[current] if v not in visited]
            if not nexts:
                if len(path) == n:
                    # Check if it closes to 0
                    if 0 in rem_adj[current]:
                        return path
                    else:
                        return None
                else:
                    return None
            # If there are multiple nexts, it's not a simple path yet or we are stuck
            # In a 2-regular graph, if we haven't visited all, and we are at a node with 2 unvisited neighbors? No, degree is 2.
            # One neighbor is previous (visited), one is next (unvisited) unless we are at the end.
            if len(nexts) > 1:
                # This implies a branch, which shouldn't happen in a 2-regular graph if we follow the path
                # Actually, in a 2-regular graph, each node has degree 2.
                # If we are at 'current', one neighbor is 'prev' (visited).
                # The other neighbor must be 'next'.
                # So 'nexts' should have exactly 1 element if we haven't closed the loop.
                # If 'nexts' has 2 elements, it means 'prev' was not in rem_adj[current]? No, we came from prev.
                # So this check is redundant if we track prev, but let's just pick the first one.
                pass
            
            next_node = nexts[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        # After n steps, we should be back at 0?
        # The path has n nodes.
        # We need to ensure the last node is connected to 0.
        if 0 in rem_adj[current]:
            return path
        return None

    # Main search
    best_cycles = []
    start_time = time.time()
    
    for trial in range(200):
        if time.time() - start_time > 55:
            break
            
        random.seed(trial)
        
        # Step 1: Find C1
        c1 = find_ham_cycle(set())
        if c1 is None:
            continue
            
        edges_c1 = set(get_cycle_edges(c1))
        
        # Step 2: Find C2 such that remainder is Hamiltonian
        found_c2_c3 = False
        for attempt in range(50):
            c2 = find_ham_cycle(edges_c1, max_attempts=10)
            if c2 is None:
                continue
            
            edges_c2 = set(get_cycle_edges(c2))
            if edges_c2 & edges_c1:
                continue
                
            used_edges = edges_c1 | edges_c2
            c3 = check_remainder_is_hamiltonian(used_edges)
            
            if c3 is not None:
                # Verify C3 is valid
                if is_hamiltonian_cycle(c3, used_edges):
                    best_cycles = [c1, c2, c3]
                    found_c2_c3 = True
                    break
        
        if found_c2_c3:
            break

    # If we didn't find 3, try to find as many as possible with the old method as fallback
    if len(best_cycles) < 3:
        used_edges = set()
        cycles = []
        for _ in range(3):
            c = find_ham_cycle(used_edges)
            if c is None:
                break
            ce = set(get_cycle_edges(c))
            if ce & used_edges:
                break
            used_edges |= ce
            cycles.append(c)
        if len(cycles) > len(best_cycles):
            best_cycles = cycles

    return [list(c) for c in best_cycles]
# EVOLVE-BLOCK-END
