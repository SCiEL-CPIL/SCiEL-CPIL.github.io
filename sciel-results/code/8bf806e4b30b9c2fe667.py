"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = 75  # safety margin before 90s kill

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _find_edge_disjoint_cycles(S, adj, max_cycles, deadline):
    """
    Find edge-disjoint Hamiltonian cycles on vertex set S.
    Uses randomized restarts with DFS backtracking.
    """
    S_set = frozenset(S)
    S_list = sorted(S)
    k = len(S_list)
    
    if k < 3:
        return []
    
    # Precompute adjacency within S
    adj_S = {}
    for v in S_list:
        adj_S[v] = adj[v] & S_set
    
    # Check minimum degree - if any vertex has degree < 2 in S, no Hamiltonian cycle exists
    for v in S_list:
        if len(adj_S[v]) < 2:
            return []
    
    used_edges = set()
    cycles = []
    
    # Try multiple random orderings of neighbor exploration
    import random
    rng = random.Random(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        if len(cycles) >= max_cycles:
            break
        
        # Shuffle neighbor orderings for diversity
        shuffled_adj = {}
        for v in S_list:
            neighbors = list(adj_S[v])
            rng.shuffle(neighbors)
            shuffled_adj[v] = neighbors
        
        # DFS to find a Hamiltonian cycle
        start = S_list[0]
        
        def dfs(path, visited, local_used):
            if time.time() > deadline:
                return None
            
            if len(path) == k:
                last = path[-1]
                ek = _edge_key(last, start)
                if start in adj_S[last] and ek not in local_used and ek not in used_edges:
                    return path
                return None
            
            for neighbor in shuffled_adj[path[-1]]:
                if neighbor not in visited:
                    ek = _edge_key(path[-1], neighbor)
                    if ek not in used_edges and ek not in local_used:
                        visited.add(neighbor)
                        local_used.add(ek)
                        result = dfs(path + [neighbor], visited, local_used)
                        if result is not None:
                            return result
                        local_used.discard(ek)
                        visited.discard(neighbor)
            return None
        
        visited = {start}
        local_used = set()
        result = dfs([start], visited, local_used)
        
        if result is not None:
            # Compute edges used in this cycle
            cycle_edges = set()
            for i in range(k):
                u = result[i]
                v = result[(i + 1) % k]
                cycle_edges.add(_edge_key(u, v))
            
            # Check no conflict with already used edges
            if not cycle_edges & used_edges:
                cycles.append(result)
                used_edges |= cycle_edges
    
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    adj = _build_adjacency()
    n = P['n']
    
    best_cycles = None
    best_count = 0
    
    # Try different vertex set sizes
    # For size s, max cycles = min(10, 70 // s)
    # We want to maximize count, so try small sizes first
    
    all_vertices = list(range(n))
    
    # Strategy: try all triangles first (fast), then 4-cycles, etc.
    # But also try larger sets if we can't get 10 small ones
    
    # First pass: try to find 10 cycles with small vertex sets
    for s in [3, 4, 5, 6, 7]:
        if time.time() > deadline - 5:
            break
        
        max_possible = min(10, 70 // s)
        if max_possible <= best_count:
            continue
        
        # Generate candidate vertex sets
        if s == 3:
            candidates = list(itertools.combinations(all_vertices, s))
        elif s == 4:
            candidates = list(itertools.combinations(all_vertices, s))
        elif s == 5:
            candidates = list(itertools.combinations(all_vertices, s))
        elif s == 6:
            candidates = list(itertools.combinations(all_vertices, s))[:5000]
        else:
            candidates = list(itertools.combinations(all_vertices, s))[:2000]
        
        for candidate in candidates:
            if time.time() > deadline - 3:
                break
            
            S = list(candidate)
            cycles = _find_edge_disjoint_cycles(S, adj, max_possible, deadline)
            
            if cycles is not None and len(cycles) > best_count:
                best_cycles = [list(c) for c in cycles]
                best_count = len(cycles)
                
                if best_count >= 10:
                    break
        
        if best_count >= 10:
            break
    
    # If we didn't get 10, try larger sets with fewer cycles
    if best_count < 10:
        for s in [8, 9, 10, 11, 12, 14, 15, 17, 18, 21]:
            if time.time() > deadline - 5:
                break
            
            max_possible = min(10, 70 // s)
            if max_possible <= best_count:
                continue
            
            if s == n:
                candidates = [tuple(all_vertices)]
            else:
                candidates = list(itertools.combinations(all_vertices, s))[:200]
            
            for candidate in candidates:
                if time.time() > deadline - 3:
                    break
                
                S = list(candidate)
                cycles = _find_edge_disjoint_cycles(S, adj, max_possible, deadline)
                
                if cycles is not None and len(cycles) > best_count:
                    best_cycles = [list(c) for c in cycles]
                    best_count = len(cycles)
            
            if best_count >= 10:
                break
    
    # Fallback: find any single cycle
    if best_cycles is None or len(best_cycles) == 0:
        for u, v in P['edges']:
            if time.time() > deadline:
                break
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    best_cycles = [[u, v, w]]
                    best_count = 1
                    break
            if best_cycles:
                break
    
    if best_cycles is None or len(best_cycles) == 0:
        return [list(range(P['n']))]
    
    # Convert types
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result

# EVOLVE-BLOCK-END
