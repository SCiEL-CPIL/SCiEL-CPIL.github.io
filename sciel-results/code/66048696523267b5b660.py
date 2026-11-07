# EVOLVE-BLOCK-START
import itertools
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE = 85  # safety margin before 90s kill

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_set(vertices):
    """Return set of edges (as frozensets) for a cycle given vertex list."""
    edges = set()
    n = len(vertices)
    for i in range(n):
        u = vertices[i]
        v = vertices[(i + 1) % n]
        edges.add(frozenset([u, v]))
    return edges

def _find_cycles_on_vertex_set(S, adj, max_cycles, deadline):
    """
    Find edge-disjoint cycles all using exactly vertex set S.
    Returns list of cycles (each a list of vertices).
    """
    S_set = frozenset(S)
    S_list = list(S)
    k = len(S_list)
    
    # Precompute: for each vertex in S, neighbors in S
    adj_S = {}
    for v in S_list:
        adj_S[v] = adj[v] & S_set
        
    # Check min degree
    min_deg = min(len(adj_S[v]) for v in S_list)
    if min_deg < 2:
        return []
        
    theoretical_max = min(max_cycles, min_deg // 2)
    if theoretical_max == 0:
        return []

    used_edges = set()
    cycles = []
    
    def backtrack(used_edges, cycles_found):
        if time.time() > deadline:
            return cycles_found
        
        if len(cycles_found) >= theoretical_max:
            return cycles_found
        
        # Try to find another Hamiltonian cycle on S avoiding used_edges
        start = S_list[0]
        
        def dfs_cycle(path, visited, local_used_edges):
            if time.time() > deadline:
                return None
            
            if len(path) == k:
                last = path[-1]
                if start in adj_S[last]:
                    edge = frozenset([last, start])
                    if edge not in local_used_edges and edge not in used_edges:
                        new_used = local_used_edges | {edge}
                        return (path + [start], new_used)
                return None
            
            # Randomize neighbors to escape local minima
            neighbors = list(adj_S[path[-1]])
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    edge = frozenset([path[-1], neighbor])
                    if edge not in used_edges and edge not in local_used_edges:
                        visited.add(neighbor)
                        local_used_edges.add(edge)
                        result = dfs_cycle(path + [neighbor], visited, local_used_edges)
                        if result is not None:
                            return result
                        local_used_edges.discard(edge)
                        visited.discard(neighbor)
            return None
        
        # Try multiple restarts with different random seeds/orders if first fails?
        # For now, just one attempt with randomized DFS
        visited = {start}
        local_used = set()
        result = dfs_cycle([start], visited, local_used)
        
        if result is not None:
            cycle, new_used = result
            cycle = cycle[:-1]
            cycles_found.append(cycle)
            return backtrack(used_edges | new_used, cycles_found)
        
        return cycles_found
    
    result = backtrack(set(), [])
    return result

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    adj = _build_adjacency()
    n = P['n']
    
    best_cycles = None
    best_count = 0
    
    # Calculate degrees
    degrees = [len(adj[v]) for v in range(n)]
    
    # Sort vertices by degree descending
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # Candidate sets: 
    # 1. Top k vertices
    # 2. Random subsets biased towards high degree
    
    all_vertices = list(range(n))
    
    # Try sizes from 7 to 15 (smaller sizes have lower theoretical max, larger might be too sparse)
    # Theoretical max for size s is bounded by min_deg//2.
    # In K_s, min_deg is s-1. Max cycles is (s-1)//2.
    # s=7 -> 3, s=8 -> 3, s=9 -> 4, s=10 -> 4, s=11 -> 5...
    
    # We want to find at least 4. So s must be at least 9 (if dense).
    # Let's try s from 9 down to 7, and also larger if possible.
    
    sizes_to_try = [9, 10, 11, 12, 8, 7, 13, 14, 15]
    
    for s in sizes_to_try:
        if time.time() > deadline - 5:
            break
            
        # Generate candidates
        candidates = set()
        
        # Candidate 1: Top s vertices
        candidates.add(tuple(sorted_vertices[:s]))
        
        # Candidate 2: Random subsets
        num_random = 20 if s <= 10 else 10
        for _ in range(num_random):
            # Pick some high degree vertices, fill rest randomly
            core_size = max(3, s // 2)
            core = random.sample(sorted_vertices[:10], core_size)
            rest = random.sample([v for v in all_vertices if v not in core], s - core_size)
            candidate = tuple(core + rest)
            candidates.add(candidate)
            
        for candidate in candidates:
            if time.time() > deadline - 2:
                break
                
            S = list(candidate)
            # Quick check: min degree in induced subgraph
            S_set = set(S)
            min_deg = min(len(adj[v] & S_set) for v in S)
            
            # Theoretical max cycles
            max_possible = min(10, min_deg // 2)
            
            if max_possible <= best_count:
                continue
                
            cycles = _find_cycles_on_vertex_set(S, adj, max_possible, deadline)
            
            if cycles is not None and len(cycles) > best_count:
                best_cycles = [list(c) for c in cycles]
                best_count = len(cycles)
                
                if best_count >= 10:
                    break
        
        if best_count >= 10:
            break
            
    # Fallback if no good set found
    if best_cycles is None or len(best_cycles) == 0:
        # Try to find any single cycle (triangle)
        # This should always be found
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
        
    # Validate and convert types
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
        
    return result

# EVOLVE-BLOCK-END
