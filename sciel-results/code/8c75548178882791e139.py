"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
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

    # Hardcoded valid partition of size 4 found via manual analysis
    # P1: {1, 6, 7, 10} (Independent)
    # P2: {4, 8, 11, 12} (Independent)
    # P3: {0, 2, 3, 5} (Clique)
    # P4: {9, 13} (Independent)
    
    hardcoded_partition = [
        [1, 6, 7, 10],
        [4, 8, 11, 12],
        [0, 2, 3, 5],
        [9, 13]
    ]
    
    # Verify hardcoded partition
    all_vertices = set(range(n))
    used_vertices = set()
    valid = True
    for part in hardcoded_partition:
        for v in part:
            if v in used_vertices:
                valid = False
            used_vertices.add(v)
        if not is_valid_part(part):
            valid = False
            
    if valid and used_vertices == all_vertices:
        return hardcoded_partition

    # Fallback solver if hardcoded fails (should not happen for fixed P)
    deadline = time.time() + 55.0
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
        if part_count >= best_count:
            return
        
        # Heuristic: pick vertex with fewest placements
        def count_placements(v):
            count = 0
            for part in current_parts:
                trial = part + [v]
                if is_valid_part(trial):
                    count += 1
            return count + 1

        min_v = min(remaining, key=count_placements)
        
        for i, part in enumerate(current_parts):
            trial = part + [min_v]
            if is_valid_part(trial):
                new_remaining = list(remaining)
                new_remaining.remove(min_v)
                new_parts = [list(p) for p in current_parts]
                new_parts[i] = trial
                search(new_remaining, new_parts, part_count)
        
        new_remaining = list(remaining)
        new_remaining.remove(min_v)
        new_parts = [list(p) for p in current_parts]
        new_parts.append([min_v])
        search(new_remaining, new_parts, part_count + 1)

    search(list(range(n)), [], 0)
    
    if best_parts is None:
        # Greedy fallback
        parts = []
        remaining = set(range(n))
        while remaining:
            v = max(remaining, key=lambda x: sum(adj[x][y] for y in remaining if y != x))
            best_part = [v]
            for u in remaining:
                if u == v: continue
                trial = best_part + [u]
                if is_valid_part(trial):
                    best_part = trial
            parts.append(best_part)
            remaining -= set(best_part)
        best_parts = parts

    return best_parts

# EVOLVE-BLOCK-END
