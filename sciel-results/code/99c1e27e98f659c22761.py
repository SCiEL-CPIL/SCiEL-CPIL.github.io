import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    def solve_instance(n, p, seed):
        # Generate graph
        rng = random.Random(seed)
        # Adjacency matrix as list of bitsets (integers)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        
        # Find all cliques of size 3 to K_MAX
        # K_MAX depends on n
        k_max = 7 if n <= 32 else 5
        
        cliques = []
        
        # Helper to check if a tuple is a clique
        def is_clique(vertices):
            # vertices is a tuple of ints
            if len(vertices) < 2:
                return True
            # Use bitsets for efficiency
            # For each v in vertices, all other vertices must be in adj[v]
            for i, v in enumerate(vertices):
                # Create mask of other vertices
                mask = 0
                for j, u in enumerate(vertices):
                    if i != j:
                        mask |= (1 << u)
                if (adj[v] & mask) != mask:
                    return False
            return True

        # Generate cliques using combinations
        # This is O(n^k). For n=64, k=5, it's ~7.6M iterations.
        # is_clique is fast with bitsets.
        
        for k in range(3, k_max + 1):
            # If time is running out, stop generating
            if time.time() - start_time > time_limit - 10:
                break
            
            for comb in combinations(range(n), k):
                if is_clique(comb):
                    cliques.append(comb)
        
        # Sort cliques by size descending
        cliques.sort(key=len, reverse=True)
        
        # Greedy selection
        # covered_edges is a set of frozensets or just a set of (min, max) tuples
        covered = set()
        partition = []
        
        for c in cliques:
            # Check if all edges in c are uncovered
            c_edges = set()
            valid = True
            for u, v in combinations(c, 2):
                e = (min(u, v), max(u, v))
                if e in covered:
                    valid = False
                    break
                c_edges.add(e)
            
            if valid:
                partition.append(list(c))
                covered.update(c_edges)
        
        # Add remaining edges
        for u, v in edges:
            e = (min(u, v), max(u, v))
            if e not in covered:
                partition.append([u, v])
        
        return partition

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() - start_time > time_limit - 5:
            # Fallback to trivial solution if running out of time
            # Just return edges for the remaining instances
            # But we need to handle the specific instance logic
            pass
        
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
