def _check_girth_circulant(n, S, min_girth):
    """Check if circulant graph C_n(S) has girth >= min_girth."""
    # Check if graph is bipartite
    # Bipartite if all steps in S are odd.
    is_bipartite = all(s % 2 == 1 for s in S)
    
    def has_cycle(L):
        # DFS from 0
        # visited: set of nodes
        # We use a list for the path to allow backtracking
        # But set is faster for lookup.
        
        # To speed up, we can use a tuple of visited? No, set is fine for small L.
        
        # Optimization: If is_bipartite and L is odd, return False immediately.
        if is_bipartite and L % 2 == 1:
            return False
            
        def dfs(current, depth, visited):
            if depth == L:
                return current == 0
            for s in S:
                next_val = (current + s) % n
                if next_val in visited:
                    continue
                if next_val == 0:
                    if depth + 1 == L:
                        return True
                    else:
                        continue # Can't close early
                visited.add(next_val)
                if dfs(next_val, depth + 1, visited):
                    return True
                visited.remove(next_val)
            return False
        
        return dfs(0, 0, {0})

    for L in range(3, min_girth):
        if has_cycle(L):
            return False
            
    return True
