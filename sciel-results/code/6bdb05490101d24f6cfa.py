"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85.0
    
    # Build adjacency matrix and edge set
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    def is_clique(vertices):
        if len(vertices) <= 1:
            return True
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if not adj[vertices[i]][vertices[j]]:
                    return False
        return True
    
    def is_independent(vertices):
        if len(vertices) <= 1:
            return True
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if adj[vertices[i]][vertices[j]]:
                    return False
        return True
    
    def is_valid_part(vertices):
        return is_clique(vertices) or is_independent(vertices)
    
    # Try to find a partition with k parts using backtracking
    def solve_k(k):
        vertices = list(range(n))
        
        def backtrack(remaining, parts):
            if time.time() > deadline:
                return None
            if not remaining:
                return parts
            if len(parts) + len(remaining) < k:
                return None
            
            # Pick the vertex with fewest options
            best_v = None
            best_options = None
            for v in remaining:
                options = 0
                for part in parts:
                    if is_valid_part(part + [v]):
                        options += 1
                if not parts or is_valid_part([v]):
                    options += 1  # new part
                if best_options is None or options < best_options:
                    best_options = options
                    best_v = v
            
            v = best_v
            remaining.remove(v)
            
            # Try adding to existing parts
            for i, part in enumerate(parts):
                if is_valid_part(part + [v]):
                    parts[i].append(v)
                    result = backtrack(remaining, parts)
                    if result is not None:
                        return result
                    parts[i].pop()
            
            # Try new part
            if len(parts) < k:
                parts.append([v])
                result = backtrack(remaining, parts)
                if result is not None:
                    return result
                parts.pop()
            
            remaining.append(v)
            return None
        
        return backtrack(vertices, [])
    
    # Try increasing k from 2
    for k in range(2, n + 1):
        result = solve_k(k)
        if result is not None:
            # Convert to proper lists of ints
            return [list(part) for part in result]
    
    # Fallback: trivial partition (each vertex its own part)
    return [[v] for v in range(n)]

# EVOLVE-BLOCK-END
