import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Known good solution: two K_{3,3} components, cycle lengths {4,6}
    def make_k33(offset):
        edges = []
        for u in range(3):
            for v in range(3):
                edges.append([offset + u, offset + 3 + v])
        return edges
    
    best_edges = make_k33(0) + make_k33(6)
    best_raw = 2
    
    # Try to find a graph with raw value 1
    # Search space is too large for exhaustive, so try targeted constructions
    
    # Attempt 1: Try to find a cubic bipartite graph on 12 vertices
    # where all cycles are the same length.
    # We'll use a randomized approach with verification.
    
    import random
    random.seed(42)
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(12):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    def count_distinct_cycle_lengths(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all simple cycles using DFS
        cycles = []
        
        def dfs(start, current, visited, path):
            if time.time() > deadline:
                return
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 3:
                    cycles.append(len(path))
                elif neighbor not in visited and neighbor > start:
                    # Only explore neighbors > start to avoid duplicates
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)
        
        for start in range(12):
            visited = {start}
            dfs(start, start, visited, [start])
        
        if not cycles:
            return 0, set()
        return len(set(cycles)), set(cycles)
    
    def is_valid(edges):
        # Check no loops, no duplicates
        if len(edges) != len(set((min(u,v), max(u,v)) for u,v in edges)):
            return False
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= 12 or v < 0 or v >= 12:
                return False
        
        # Check degree >= 3
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for i in range(12):
            if len(adj[i]) < 3:
                return False
        
        # Check triangle-free
        for u in range(12):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
        
        # Check has a cycle
        raw, lengths = count_distinct_cycle_lengths(edges)
        if raw == 0:
            return False
        
        return True
    
    # Try random cubic bipartite graphs
    # Bipartition: {0,1,2,3,4,5} and {6,7,8,9,10,11}
    # Each vertex in left has degree 3, each in right has degree 3
    
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 5000:
        attempts += 1
        
        # Random cubic bipartite graph
        # Left: 0-5, Right: 6-11
        # Each left vertex connects to 3 random right vertices
        # Each right vertex connects to 3 random left vertices
        
        left = list(range(6))
        right = list(range(6, 12))
        
        # Generate random permutation-based construction
        # Each left vertex picks 3 right vertices
        # Need to ensure each right vertex has degree exactly 3
        
        # Use random matching approach
        adj = {i: set() for i in range(12)}
        
        # Try to build a random 3-regular bipartite graph
        # Using the configuration model or random sampling
        
        # Simple approach: random assignment
        for _ in range(100):
            if time.time() > deadline:
                break
            adj = {i: set() for i in range(12)}
            
            # Each left vertex chooses 3 right vertices
            left_choices = {}
            for u in left:
                left_choices[u] = set(random.sample(right, 3))
            
            # Check if each right vertex has degree 3
            right_deg = {v: 0 for v in right}
            for u in left:
                for v in left_choices[u]:
                    right_deg[v] += 1
            
            if all(d == 3 for d in right_deg.values()):
                edges = []
                for u in left:
                    for v in left_choices[u]:
                        edges.append([u, v])
                
                if is_valid(edges):
                    raw, lengths = count_distinct_cycle_lengths(edges)
                    if raw < best_raw:
                        best_raw = raw
                        best_edges = edges
                        if raw == 1:
                            break
                break
        
        if best_raw == 1:
            break
    
    # Convert to proper types
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
