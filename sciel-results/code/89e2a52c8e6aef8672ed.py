"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build adjacency and edge lookup
    adj = {i: [] for i in range(n)}
    edge_set = set()
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        edge_set.add((min(u, v), max(u, v)))
    
    edge_id_map = {}
    for idx, (u, v) in enumerate(edges):
        edge_id_map[(min(u, v), max(u, v))] = idx
    
    def get_eid(u, v):
        return edge_id_map.get((min(u, v), max(u, v)), -1)
    
    # Precompute neighbor lists
    neighbors = [adj[i] for i in range(n)]
    
    deadline = time.time() + 75  # Safety margin
    
    def find_ham_cycle(used_edges, start=0, time_limit=None, seed=None, max_steps=500000):
        """Find a Hamiltonian cycle avoiding used_edges, starting from 'start'."""
        if time_limit is None:
            time_limit = deadline
        
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        visited = [False] * n
        visited[start] = True
        path = [start]
        path_edges = []
        steps = [0]
        
        def dfs(current, depth):
            steps[0] += 1
            if steps[0] > max_steps or (steps[0] % 100 == 0 and time.time() > time_limit):
                return None
            if depth == n:
                eid = get_eid(current, start)
                if eid != -1 and eid not in used_edges:
                    return path + [start], path_edges + [eid]
                return None
            
            candidates = []
            for (nb, eid) in neighbors[current]:
                if nb == start:
                    continue
                if not visited[nb] and eid not in used_edges:
                    candidates.append((nb, eid))
            
            if not candidates:
                return None
            
            # MRV heuristic: prefer vertices with fewer available unvisited neighbors
            def mrv_score(nb):
                cnt = 0
                for (nb2, eid2) in neighbors[nb]:
                    if not visited[nb2] and eid2 not in used_edges and nb2 != start:
                        cnt += 1
                return cnt
            
            scored = [(mrv_score(nb), nb, eid) for nb, eid in candidates]
            scored.sort(key=lambda x: x[0])
            
            # Group by score and shuffle within groups for diversity
            new_candidates = []
            prev_score = None
            group = []
            for s, nb, eid in scored:
                if prev_score is not None and s != prev_score:
                    rng.shuffle(group)
                    new_candidates.extend([(nb2, eid2) for _, nb2, eid2 in group])
                    group = []
                group.append((s, nb, eid))
                prev_score = s
            if group:
                rng.shuffle(group)
                new_candidates.extend([(nb2, eid2) for _, nb2, eid2 in group])
            
            for (nb, eid) in new_candidates:
                if steps[0] % 200 == 0 and time.time() > time_limit:
                    return None
                visited[nb] = True
                path.append(nb)
                path_edges.append(eid)
                
                result = dfs(nb, depth + 1)
                if result is not None:
                    return result
                
                path.pop()
                path_edges.pop()
                visited[nb] = False
            
            return None
        
        return dfs(start, 1)
    
    def validate_cycles(cycles):
        """Validate that cycles are valid and edge-disjoint."""
        if not cycles:
            return False
        used = set()
        for cycle in cycles:
            if len(cycle) != n:
                return False
            if len(set(cycle)) != n:
                return False
            for i in range(n):
                u = cycle[i]
                v = cycle[(i + 1) % n]
                eid = get_eid(u, v)
                if eid == -1:
                    return False
                if eid in used:
                    return False
                used.add(eid)
        return True
    
    # Strategy: Try multiple start vertices and orderings
    best_cycles = []
    
    # Try several deterministic start orderings
    start_orders = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        [0, 10, 20, 19, 18, 17, 16, 15, 14, 13, 12],
        [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [5, 10, 15, 0, 5, 10, 15, 0, 5, 10, 15],
        [10, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11],
        [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
    ]
    
    for order in start_orders:
        if time.time() > deadline - 2:
            break
        cycles_try = []
        used_try = set()
        for start in order:
            if time.time() > deadline - 2:
                break
            result = find_ham_cycle(used_try, start)
            if result is not None:
                cycle_path, cycle_eids = result
                cycles_try.append(cycle_path[:-1])
                used_try.update(cycle_eids)
                if len(cycles_try) >= 3:
                    break
        if len(cycles_try) > len(best_cycles):
            best_cycles = cycles_try
            if len(best_cycles) >= 3:
                break
    
    # Strategy 2: Randomized restarts if we didn't get 3
    if len(best_cycles) < 3 and time.time() < deadline - 3:
        for attempt in range(300):
            if time.time() > deadline - 3:
                break
            used2 = set()
            cycles2 = []
            starts = list(range(n))
            random.shuffle(starts)
            for start in starts:
                if time.time() > deadline - 3:
                    break
                result = find_ham_cycle(used2, start, seed=attempt * 1000 + start)
                if result is not None:
                    cycle_path, cycle_eids = result
                    cycles2.append(cycle_path[:-1])
                    used2.update(cycle_eids)
                    if len(cycles2) >= 3:
                        break
            if len(cycles2) > len(best_cycles):
                best_cycles = cycles2
                if len(best_cycles) >= 3:
                    break
    
    # Strategy 3: If we have 2 cycles, try harder for the 3rd
    if len(best_cycles) == 2 and time.time() < deadline - 1:
        used3 = set()
        for cyc in best_cycles:
            for i in range(n):
                u = cyc[i]
                v = cyc[(i + 1) % n]
                eid = get_eid(u, v)
                if eid != -1:
                    used3.add(eid)
        
        for attempt in range(500):
            if time.time() > deadline - 1:
                break
            start = random.randint(0, n - 1)
            result = find_ham_cycle(used3, start, seed=100000 + attempt)
            if result is not None:
                cycle_path, cycle_eids = result
                best_cycles.append(cycle_path[:-1])
                break
    
    # Ensure at least one valid cycle
    if not best_cycles:
        for start in range(n):
            result = find_ham_cycle(set(), start)
            if result is not None:
                best_cycles = [result[0][:-1]]
                break
        if not best_cycles:
            # Try a simple fallback: use a known cycle structure
            # The graph is dense, so a simple cycle should exist
            best_cycles = [list(range(n))]
    
    # Final validation
    if not validate_cycles(best_cycles):
        # Fallback to a known valid single cycle
        best_cycles = [list(range(n))]
        if not validate_cycles(best_cycles):
            # Try to find any valid cycle
            for start in range(n):
                result = find_ham_cycle(set(), start)
                if result is not None:
                    best_cycles = [result[0][:-1]]
                    break
    
    # Validate and convert to proper types
    validated = []
    for cycle in best_cycles:
        cycle = [int(v) for v in cycle]
        validated.append(list(cycle))
    
    return validated
# EVOLVE-BLOCK-END
