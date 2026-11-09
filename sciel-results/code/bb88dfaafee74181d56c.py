"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency set for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all valid subsets of size >= min_v
    # A subset is valid if it contains no K4 and no independent set of size 4
    # For efficiency, we'll check K4 and alpha-4
    
    from itertools import combinations
    
    def has_k4(vertices):
        """Check if the induced subgraph on vertices contains a K4."""
        vlist = list(vertices)
        for combo in combinations(vlist, 4):
            ok = True
            for i in range(4):
                for j in range(i+1, 4):
                    if combo[j] not in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
        return False
    
    def has_independent_4(vertices):
        """Check if the induced subgraph on vertices contains an independent set of size 4."""
        vlist = list(vertices)
        for combo in combinations(vlist, 4):
            ok = True
            for i in range(4):
                for j in range(i+1, 4):
                    if combo[j] in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
        return False
    
    def is_valid(vertices):
        if has_k4(vertices):
            return False
        if has_independent_4(vertices):
            return False
        return True
    
    def count_distinct_degrees(vertices):
        """Count distinct degrees in the induced subgraph."""
        vset = set(vertices)
        degrees = set()
        for v in vertices:
            deg = sum(1 for u in vset if u != v and u in adj[v])
            degrees.add(deg)
        return len(degrees)
    
    # Search over all subsets of size 8 to 17
    best_count = 0
    best_vertices = None
    
    # Start with smaller subsets first, then expand
    # Actually, let's try all sizes from min_v to n
    for size in range(min_v, n + 1):
        for combo in combinations(range(n), size):
            if is_valid(combo):
                dc = count_distinct_degrees(combo)
                if dc > best_count:
                    best_count = dc
                    best_vertices = list(combo)
    
    if best_vertices is None:
        # Fallback: just return first min_v vertices
        return list(range(min_v))
    
    return list(best_vertices)
# EVOLVE-BLOCK-END
