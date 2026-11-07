    def _can_match(r, L, n=12):
        # Values are r+1 to r+L
        # Map value v to index v - (r+1)
        # But we only care about values that are multiples of some i.
        # Let's just build the graph.
        
        # adj[i] = list of values in (r, r+L] divisible by i
        adj = [set() for _ in range(n)]
        for i in range(1, n+1):
            # First multiple of i > r
            start = ((r // i) + 1) * i
            if start > r + L:
                return False
            v = start
            while v <= r + L:
                adj[i-1].add(v)
                v += i
        
        # Bipartite matching
        # match_val_to_idx: value -> index i
        match_val_to_idx = {}
        
        def dfs(u, visited):
            for v in adj[u]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_val_to_idx or dfs(match_val_to_idx[v], visited):
                    match_val_to_idx[v] = u
                    return True
            return False
        
        for u in range(n):
            visited = set()
            if not dfs(u, visited):
                return False
        return True
    