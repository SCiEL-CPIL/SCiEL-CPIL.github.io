"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    # Precompute adjacency
    adj = [set() for _ in range(n)]
    for (u, v) in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 55  # safety margin
    
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if vertices[j] not in adj[vertices[i]]:
                    return False
        return True
    
    def is_independent(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if vertices[j] in adj[vertices[i]]:
                    return False
        return True
    
    # Try to find minimum number of parts
    best_parts = None
    best_count = n + 1  # worst case: each vertex its own part
    
    # Use backtracking
    def backtrack(remaining, parts, part_count):
        nonlocal best_parts, best_count, deadline
        
        if time.time() > deadline:
            return
        
        if part_count >= best_count:
            return
        
        if not remaining:
            if part_count < best_count:
                best_count = part_count
                best_parts = [list(p) for p in parts]
            return
        
        # Lower bound: at least ceil(len(remaining) / max_possible_part_size)
        # Simple bound: at least 1 part needed
        if part_count + 1 >= best_count:
            return
        
        # Pick the first remaining vertex (or use heuristic)
        v = remaining[0]
        rest = remaining[1:]
        
        # Try adding v to each existing part
        for i, part in enumerate(parts):
            new_part = list(part) + [v]
            if is_clique(new_part) or is_independent(new_part):
                new_parts = [list(p) for p in parts]
                new_parts[i] = new_part
                backtrack(rest, new_parts, part_count)
        
        # Try starting a new part with v
        new_parts = [list(p) for p in parts] + [[v]]
        backtrack(rest, new_parts, part_count + 1)
    
    backtrack(list(range(n)), [], 0)
    
    if best_parts is None:
        # Fallback: greedy
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                if is_clique(trial) or is_independent(trial):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        best_parts = parts
    
    # Ensure proper types
    result = []
    for part in best_parts:
        result.append([int(v) for v in part])
    return result

# EVOLVE-BLOCK-END
