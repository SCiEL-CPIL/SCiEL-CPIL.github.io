import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = 21
    edges = P['edges']
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
        adj[u].add(v)
        adj[v].add(u)
    
    best_cycles = []
    best_count = 0
    
    # Try different vertex subset sizes
    # For a subset S of size k, we need edge-disjoint Hamiltonian cycles on S
    # Each Hamiltonian cycle on S uses k edges
    # Max cycles = (number of edges in induced subgraph) / k
    
    # Start with full vertex set
    all_vertices = set(range(n))
    
    def find_hamiltonian_cycle(vertices, used_edges, adj):
        """Find a Hamiltonian cycle on the given vertex set avoiding used_edges."""
        vertices = list(vertices)
        k = len(vertices)
        if k < 3:
            return None
        
        # Try to find a Hamiltonian cycle using backtracking
        # Fix first vertex to reduce symmetry
        start = vertices[0]
        remaining = set(vertices) - {start}
        path = [start]
        
        def backtrack(path, remaining):
            if len(path) == k:
                # Check if last vertex connects to first
                last = path[-1]
                edge = (min(last, start), max(last, start))
                if edge not in used_edges:
                    return path[:]
                return None
            
            last = path[-1]
            # Try neighbors in deterministic order for reproducibility
            for v in sorted(remaining):
                if v in adj[last]:
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
        
        result = backtrack(path, remaining)
        if result:
            # Verify it's a valid cycle
            for i in range(k):
                u = result[i]
                v = result[(i+1) % k]
                edge = (min(u,v), max(u,v))
                if edge not in adj[u]:
                    return None
            return result
        return None
    
    def try_vertex_subset(vertices):
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
                    edge = (min(u,v), max(u,v))
                    induced_edges.add(edge)
        
        max_possible = len(induced_edges) // k
        if max_possible <= best_count:
            return
        
        used_edges = set()
        cycles = []
        
        # Try to find cycles greedily
        for attempt in range(max_possible):
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(vertices, used_edges, adj)
            if cycle is None:
                break
            # Add edges to used
            for i in range(k):
                u = cycle[i]
                v = cycle[(i+1) % k]
                edge = (min(u,v), max(u,v))
                used_edges.add(edge)
            cycles.append(cycle)
        
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = [list(c) for c in cycles]
    
    # Try full vertex set
    try_vertex_subset(all_vertices)
    
    # Try subsets of size 20, 19, etc.
    # For smaller subsets, we might get more cycles
    for k in range(20, 2, -1):
        if time.time() > deadline:
            break
        
        # Try random subsets of size k
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            subset = set(random.sample(range(n), k))
            try_vertex_subset(subset)
    
    # Also try specific structured subsets
    # Remove one vertex at a time
    for removed in range(n):
        if time.time() > deadline:
            break
        subset = all_vertices - {removed}
        try_vertex_subset(subset)
    
    # Return best result
    if best_cycles:
        # Ensure types are correct
        result = []
        for cycle in best_cycles:
            result.append([int(v) for v in cycle])
        return result
    else:
        # Fallback: single cycle using all vertices
        return [list(range(n))]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
