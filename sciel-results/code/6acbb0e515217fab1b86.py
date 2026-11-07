"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time
from itertools import combinations

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    # Precompute adjacency for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if (vertices[i], vertices[j]) not in edge_set:
                    return False
        return True
    
    def is_independent(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if (vertices[i], vertices[j]) in edge_set:
                    return False
        return True
    
    def is_valid_part(vertices):
        return is_clique(vertices) or is_independent(vertices)
    
    best_parts = None
    deadline = time.time() + 50  # Safety margin
    
    # Start with greedy solution
    greedy_parts = []
    for v in range(n):
        placed = False
        for part in greedy_parts:
            if is_valid_part(part + [v]):
                part.append(v)
                placed = True
                break
        if not placed:
            greedy_parts.append([v])
    
    best_parts = [list(p) for p in greedy_parts]
    best_count = len(best_parts)
    
    # Try to improve by local search: try to merge parts
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = [list(p) for p in best_parts]
        for i in range(len(current)):
            for j in range(i+1, len(current)):
                # Try merging current[i] and current[j]
                merged = current[i] + current[j]
                if is_valid_part(merged):
                    new_parts = [p for k, p in enumerate(current) if k != i and k != j]
                    new_parts.append(merged)
                    if len(new_parts) < len(best_parts):
                        best_parts = new_parts
                        best_count = len(best_parts)
                        improved = True
                        break
            if improved:
                break
    
    # Try to improve by moving vertices between parts
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = [list(p) for p in best_parts]
        for i in range(len(current)):
            for v in list(current[i]):
                # Try moving v to another part
                for j in range(len(current)):
                    if i == j:
                        continue
                    # Check if v can be added to part j
                    trial = current[j] + [v]
                    if is_valid_part(trial):
                        # Check if removing v from part i leaves it valid
                        remaining = [x for x in current[i] if x != v]
                        if remaining and is_valid_part(remaining):
                            # Do the move
                            current[i].remove(v)
                            current[j].append(v)
                            # Remove empty parts
                            current = [p for p in current if p]
                            if len(current) < len(best_parts):
                                best_parts = [list(p) for p in current]
                                best_count = len(best_parts)
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
    
    # Try to improve by splitting and reassigning
    if time.time() < deadline:
        for _ in range(100):
            if time.time() > deadline:
                break
            current = [list(p) for p in best_parts]
            # Pick a random part and try to split it
            if len(current) < 2:
                break
            part_idx = 0
            part = current[part_idx]
            if len(part) < 2:
                continue
            # Try different splits
            for split_size in range(1, len(part)):
                for subset in combinations(part, split_size):
                    subset_list = list(subset)
                    remaining_list = [x for x in part if x not in subset_list]
                    if not is_valid_part(subset_list) or not is_valid_part(remaining_list):
                        continue
                    # Try to place subset_list into existing parts or create new
                    new_parts = [p for k, p in enumerate(current) if k != part_idx]
                    # Try to fit subset_list into existing parts
                    placed = False
                    for p in new_parts:
                        if is_valid_part(p + subset_list):
                            p.extend(subset_list)
                            placed = True
                            break
                    if not placed:
                        new_parts.append(subset_list)
                    # Add remaining
                    placed_rem = False
                    for p in new_parts:
                        if is_valid_part(p + remaining_list):
                            p.extend(remaining_list)
                            placed_rem = True
                            break
                    if not placed_rem:
                        new_parts.append(remaining_list)
                    if len(new_parts) < len(best_parts):
                        best_parts = [list(p) for p in new_parts]
                        best_count = len(best_parts)
                        break
            if len(best_parts) <= best_count:
                break
    
    # Ensure all parts are nonempty lists of ints
    result = []
    for part in best_parts:
        if part:
            result.append([int(v) for v in part])
    
    # Verify all vertices are included exactly once
    all_vertices = []
    for part in result:
        all_vertices.extend(part)
    if sorted(all_vertices) != list(range(n)):
        # Fallback to greedy if something went wrong
        return greedy_parts
    
    return result
# EVOLVE-BLOCK-END
