"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        u, v = e
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    # Adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edge_set:
        adj[u][v] = True
        adj[v][u] = True
    
    def is_clique(subset):
        if len(subset) <= 1:
            return True
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                if not adj[subset[i]][subset[j]]:
                    return False
        return True
    
    def is_independent(subset):
        if len(subset) <= 1:
            return True
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                if adj[subset[i]][subset[j]]:
                    return False
        return True
    
    def is_valid_part(subset):
        return is_clique(subset) or is_independent(subset)
    
    # Backtracking search for minimum cochromatic partition
    deadline = time.time() + 50.0  # Safety margin before 60s limit
    
    best_parts = None
    best_count = n + 1
    
    def search(remaining, current_parts, part_count):
        nonlocal best_parts, best_count
        
        if time.time() > deadline:
            return
        
        if not remaining:
            if part_count < best_count:
                best_count = part_count
                best_parts = [list(p) for p in current_parts]
            return
        
        # Pruning: if we can't beat best, return
        if part_count >= best_count:
            return
        
        # Try to add remaining vertices to existing parts or create new parts
        # Use heuristic: try vertices in order of fewest possible placements
        
        # Find the vertex with minimum number of valid placements
        def count_placements(v):
            count = 0
            for part in current_parts:
                trial = part + [v]
                if is_valid_part(trial):
                    count += 1
            # Plus one for creating a new part
            return count + 1
        
        # Pick vertex with fewest placements
        min_v = min(remaining, key=count_placements)
        
        # Try adding min_v to each existing part
        for i, part in enumerate(current_parts):
            trial = part + [min_v]
            if is_valid_part(trial):
                new_remaining = list(remaining)
                new_remaining.remove(min_v)
                new_parts = [list(p) for p in current_parts]
                new_parts[i] = trial
                search(new_remaining, new_parts, part_count)
        
        # Try creating a new part with min_v
        new_remaining = list(remaining)
        new_remaining.remove(min_v)
        new_parts = [list(p) for p in current_parts]
        new_parts.append([min_v])
        search(new_remaining, new_parts, part_count + 1)
    
    # Initial greedy solution for upper bound
    def greedy():
        parts = []
        remaining = set(range(n))
        while remaining:
            # Pick vertex with highest degree in remaining subgraph
            v = max(remaining, key=lambda x: sum(adj[x][y] for y in remaining if y != x))
            # Try to find a valid part containing v
            best_part = [v]
            for u in remaining:
                if u == v:
                    continue
                trial = best_part + [u]
                if is_valid_part(trial):
                    best_part = trial
            parts.append(best_part)
            remaining -= set(best_part)
        return parts
    
    initial_parts = greedy()
    best_count = len(initial_parts)
    best_parts = initial_parts
    
    # Now try to improve with backtracking
    search(list(range(n)), [], 0)
    
    if best_parts is None:
        best_parts = initial_parts
    
    # Convert to proper types
    result = []
    for part in best_parts:
        result.append(list(part))
    
    return result

# EVOLVE-BLOCK-END
