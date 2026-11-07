import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit
    
    n = 21
    edges = P['edges']
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
    
    best_cycles = []
    best_count = 0
    
    def find_hamiltonian_cycle_fast(vertices, used_edges, adj, deadline):
        """Find a Hamiltonian cycle on vertices avoiding used_edges, with heuristic ordering."""
        vertices = list(vertices)
        k = len(vertices)
        if k < 3:
            return None
        
        # Precompute available neighbors for each vertex
        avail = {}
        for v in vertices:
            avail[v] = [u for u in adj[v] if u in vertices and (min(u,v), max(u,v)) not in used_edges]
            if len(avail[v]) < 2:
                return None  # Can't form a cycle if any vertex has < 2 available edges
        
        # Sort vertices by available degree for heuristic
        start = min(vertices, key=lambda v: len(avail[v]))
        remaining = set(vertices) - {start}
        path = [start]
        
        def backtrack(path, remaining):
            nonlocal deadline
            if time.time() > deadline:
                return None
            if len(path) == k:
                last = path[-1]
                edge = (min(last, start), max(last, start))
                if edge not in used_edges:
                    return path[:]
                return None
            
            last = path[-1]
            # Heuristic: try neighbors with fewer available options first
            candidates = []
            for v in remaining:
                if v in adj[last]:
                    edge = (min(last, v), max(last, v))
                    if edge not in used_edges:
                        # Count how many options v would have after using this edge
                        future_options = 0
                        for w in avail[v]:
                            if w == last:
                                continue
                            if (min(v, w), max(v, w)) not in used_edges:
                                future_options += 1
                        candidates.append((future_options, v))
            
            candidates.sort()  # Try most constrained first
            
            for _, v in candidates:
                if time.time() > deadline:
                    return None
                edge = (min(last, v), max(last, v))
                if edge not in used_edges:
                    path.append(v)
                    remaining.remove(v)
                    result = backtrack(path, remaining)
                    if result is not None:
                        return result
                    path.pop()
                    remaining.add(v)
            return None
        
        return backtrack(path, remaining)
    
    def try_vertex_subset(vertices, deadline):
        """Try to find max edge-disjoint Hamiltonian cycles on this vertex set."""
        nonlocal best_cycles, best_count
        if time.time() > deadline:
            return
        
        vertices = frozenset(vertices)
        k = len(vertices)
        if k < 3:
            return
        
        # Count edges in induced subgraph
        induced_edges = set()
        for u in vertices:
            for v in adj[u]:
                if v in vertices:
                    edge = (min(u, v), max(u, v))
                    induced_edges.add(edge)
        
        max_possible = len(induced_edges) // k
        if max_possible <= best_count:
            return
        
        used_edges = set()
        cycles = []
        
        # Try to find cycles greedily with multiple restarts
        max_attempts = max_possible
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle_fast(vertices, used_edges, adj, deadline)
            if cycle is None:
                break
            # Add edges to used
            for i in range(k):
                u = cycle[i]
                v = cycle[(i + 1) % k]
                edge = (min(u, v), max(u, v))
                used_edges.add(edge)
            cycles.append(cycle)
        
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = [list(c) for c in cycles]
    
    all_vertices = set(range(n))
    
    # Try full vertex set
    try_vertex_subset(all_vertices, deadline)
    
    # Try all single-vertex removals
    for removed in range(n):
        if time.time() > deadline:
            break
        subset = all_vertices - {removed}
        try_vertex_subset(subset, deadline)
    
    # Try random subsets of various sizes
    random.seed(42)
    for k in range(18, 20):
        for _ in range(30):
            if time.time() > deadline:
                break
            subset = set(random.sample(range(n), k))
            try_vertex_subset(subset, deadline)
    
    # Try subsets of size 20 with different random seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        random.seed(seed)
        for _ in range(10):
            if time.time() > deadline:
                break
            subset = set(random.sample(range(n), 20))
            try_vertex_subset(subset, deadline)
    
    # Return best result
    if best_cycles:
        result = []
        for cycle in best_cycles:
            result.append([int(v) for v in cycle])
        return result
    else:
        # Fallback: try to find any cycle on full vertex set
        cycle = find_hamiltonian_cycle_fast(all_vertices, set(), adj, deadline)
        if cycle:
            return [[int(v) for v in cycle]]
        return [list(range(n))]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
