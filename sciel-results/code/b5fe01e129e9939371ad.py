import time
import random
import itertools

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 55
    rng = random.Random(42)
    
    # Build adjacency structures
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    all_edges_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    def get_cycle_edges(cycle):
        es = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            es.add((min(u, v), max(u, v)))
        return es
    
    def is_valid_ham_cycle(cycle, forbidden_edges):
        if len(cycle) != n or set(cycle) != set(range(n)):
            return False
        for e in get_cycle_edges(cycle):
            if e not in all_edges_set or e in forbidden_edges:
                return False
        return True
    
    def normalize_cycle(cycle):
        """Normalize cycle representation for uniqueness checking."""
        min_idx = cycle.index(min(cycle))
        norm1 = cycle[min_idx:] + cycle[:min_idx]
        norm2 = [norm1[0]] + norm1[1:][::-1]
        return tuple(min(norm1, norm2))
    
    def find_ham_cycle_fast(adj_set, forbidden_edges, start=0, time_limit=2.0):
        """Find a Hamiltonian cycle using DFS with pruning."""
        start_time = time.time()
        path = [start]
        visited = {start}
        
        # Precompute adjacency lists for faster iteration
        adj_list = {i: sorted(adj_set[i] - {i}) for i in range(n)}
        # Filter out forbidden edges
        for u in adj_list:
            adj_list[u] = [v for v in adj_list[u] if (min(u, v), max(u, v)) not in forbidden_edges]
        
        def backtrack():
            if time.time() - start_time > time_limit:
                return False
            if len(path) == n:
                last = path[-1]
                if start in adj_set[last] and (min(last, start), max(last, start)) not in forbidden_edges:
                    return True
                return False
            
            last = path[-1]
            # Try neighbors in a smart order
            neighbors = adj_list[last]
            # Prefer neighbors that can still connect back to start if we're near the end
            if len(path) >= n - 2:
                neighbors.sort(key=lambda x: 0 if x == start else 1)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    # Check if this edge is forbidden
                    ekey = (min(last, neighbor), max(last, neighbor))
                    if ekey in forbidden_edges:
                        continue
                    
                    # Pruning: if we're at the last step, neighbor must connect to start
                    if len(path) == n - 1:
                        if start not in adj_set[neighbor]:
                            continue
                        if (min(neighbor, start), max(neighbor, start)) in forbidden_edges:
                            continue
                    
                    path.append(neighbor)
                    visited.add(neighbor)
                    if backtrack():
                        return True
                    path.pop()
                    visited.remove(neighbor)
            return False
        
        if backtrack():
            return path
        return None
    
    def find_ham_cycle_randomized(adj_set, forbidden_edges, num_tries=50, time_per_try=0.5):
        """Try to find a Hamiltonian cycle with randomized neighbor ordering."""
        start_time = time.time()
        
        for _ in range(num_tries):
            if time.time() - start_time > 5.0 or time.time() > deadline:
                break
            
            # Create randomized adjacency
            adj_rand = {i: list(adj_set[i] - {i}) for i in range(n)}
            for u in adj_rand:
                adj_rand[u] = [v for v in adj_rand[u] if (min(u, v), max(u, v)) not in forbidden_edges]
                rng.shuffle(adj_rand[u])
            
            start = rng.randint(0, n - 1)
            path = [start]
            visited = {start}
            
            def backtrack():
                if time.time() > deadline:
                    return False
                if len(path) == n:
                    last = path[-1]
                    if start in adj_set[last] and (min(last, start), max(last, start)) not in forbidden_edges:
                        return True
                    return False
                
                last = path[-1]
                neighbors = adj_rand[last]
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        ekey = (min(last, neighbor), max(last, neighbor))
                        if ekey in forbidden_edges:
                            continue
                        
                        if len(path) == n - 1:
                            if start not in adj_set[neighbor]:
                                continue
                            if (min(neighbor, start), max(neighbor, start)) in forbidden_edges:
                                continue
                        
                        path.append(neighbor)
                        visited.add(neighbor)
                        if backtrack():
                            return True
                        path.pop()
                        visited.remove(neighbor)
                return False
            
            if backtrack():
                return path
        return None
    
    def find_diverse_ham_cycles(adj_set, forbidden_edges, count=5, max_time=10.0):
        """Find up to 'count' distinct Hamiltonian cycles."""
        cycles = []
        seen = set()
        start_time = time.time()
        
        # First try deterministic approaches with different starts
        for start in range(n):
            if time.time() - start_time > max_time or len(cycles) >= count:
                break
            
            adj_list = {i: sorted(adj_set[i] - {i}) for i in range(n)}
            for u in adj_list:
                adj_list[u] = [v for v in adj_list[u] if (min(u, v), max(u, v)) not in forbidden_edges]
            
            path = [start]
            visited = {start}
            
            def backtrack():
                if time.time() > deadline or time.time() - start_time > max_time:
                    return False
                if len(path) == n:
                    last = path[-1]
                    if start in adj_set[last] and (min(last, start), max(last, start)) not in forbidden_edges:
                        return True
                    return False
                
                last = path[-1]
                neighbors = adj_list[last]
                if len(path) >= n - 2:
                    neighbors = sorted(neighbors, key=lambda x: 0 if x == start else 1)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        ekey = (min(last, neighbor), max(last, neighbor))
                        if ekey in forbidden_edges:
                            continue
                        
                        if len(path) == n - 1:
                            if start not in adj_set[neighbor]:
                                continue
                            if (min(neighbor, start), max(neighbor, start)) in forbidden_edges:
                                continue
                        
                        path.append(neighbor)
                        visited.add(neighbor)
                        if backtrack():
                            return True
                        path.pop()
                        visited.remove(neighbor)
                return False
            
            if backtrack():
                cycle = path
                norm = normalize_cycle(cycle)
                if norm not in seen:
                    seen.add(norm)
                    cycles.append(cycle)
        
        # If not enough, try randomized
        while len(cycles) < count and time.time() - start_time < max_time:
            c = find_ham_cycle_randomized(adj_set, forbidden_edges, num_tries=20, time_per_try=0.3)
            if c:
                norm = normalize_cycle(c)
                if norm not in seen:
                    seen.add(norm)
                    cycles.append(c)
        
        return cycles
    
    def trace_2regular_cycle(rem_adj):
        """Trace a single cycle in a 2-regular graph. Returns list of vertices or None."""
        if not all(len(nei) == 2 for nei in rem_adj.values()):
            return None
        visited = set()
        curr = 0
        path = []
        while curr not in visited:
            visited.add(curr)
            path.append(curr)
            if len(path) == 1:
                next_node = rem_adj[curr][0]
            else:
                prev_node = path[-2]
                next_node = rem_adj[curr][0] if rem_adj[curr][1] == prev_node else rem_adj[curr][1]
            curr = next_node
        if len(path) == n and 0 in rem_adj[path[-1]]:
            return path
        return None
    
    # Main search: try to find 3 edge-disjoint Hamiltonian cycles
    best_result = []
    
    # Get diverse C1 candidates
    c1_candidates = find_diverse_ham_cycles(adj, set(), count=8, max_time=15.0)
    
    for c1 in c1_candidates:
        if time.time() > deadline:
            break
        used_1 = get_cycle_edges(c1)
        
        # Get diverse C2 candidates given C1
        c2_candidates = find_diverse_ham_cycles(adj, used_1, count=8, max_time=15.0)
        
        for c2 in c2_candidates:
            if time.time() > deadline:
                break
            used_2 = get_cycle_edges(c2)
            all_used = used_1 | used_2
            
            # Check if remaining edges form a single Hamiltonian cycle
            rem_adj = {i: [] for i in range(n)}
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key not in all_used:
                    rem_adj[u].append(v)
                    rem_adj[v].append(u)
            
            c3 = trace_2regular_cycle(rem_adj)
            if c3 and is_valid_ham_cycle(c3, all_used):
                result = [list(c1), list(c2), list(c3)]
                # Final validation
                vset = set(range(n))
                seen_edges = set()
                valid = True
                for c in result:
                    if set(c) != vset or len(c) != n:
                        valid = False
                        break
                    for e in get_cycle_edges(c):
                        if e in seen_edges:
                            valid = False
                            break
                        seen_edges.add(e)
                    if not valid:
                        break
                if valid:
                    best_result = result
                    break
        
        if best_result:
            break
    
    # Fallback: try to get at least 2 cycles
    if not best_result:
        c1 = find_ham_cycle_fast(adj, set(), start=0, time_limit=5.0)
        if c1 and is_valid_ham_cycle(c1, set()):
            used = get_cycle_edges(c1)
            c2 = find_ham_cycle_fast(adj, used, start=0, time_limit=5.0)
            if c2 and is_valid_ham_cycle(c2, used):
                best_result = [list(c1), list(c2)]
    
    # If still nothing, try even harder
    if not best_result:
        c1 = find_ham_cycle_randomized(adj, set(), num_tries=100, time_per_try=0.2)
        if c1 and is_valid_ham_cycle(c1, set()):
            used = get_cycle_edges(c1)
            c2 = find_ham_cycle_randomized(adj, used, num_tries=100, time_per_try=0.2)
            if c2 and is_valid_ham_cycle(c2, used):
                best_result = [list(c1), list(c2)]
    
    return [list(map(int, c)) for c in best_result]
