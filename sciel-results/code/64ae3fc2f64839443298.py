    def check_add_edge(u, v, adj):
        # Check triples containing u
        for a in range(n):
            if a == u: continue
            for b in range(a+1, n):
                if b == u: continue
                # Triple is (u, a, b) sorted
                # We need i<j<k
                # Let's just get the 3 vertices and sort them
                # Or simpler:
                # The common neighbors of (u, a, b) is adj[u] & adj[a] & adj[b]
                # We don't need to sort if we just compute the intersection.
                # The condition is symmetric.
                common = adj[u] & adj[a] & adj[b]
                if common.bit_count() >= 3:
                    return False
        
        # Check triples containing v
        # But we already checked triples containing both u and v in the first loop?
        # No, the first loop checks triples (u, a, b).
        # If a or b is v, it's (u, v, x).
        # So triples with both u and v are covered in the first loop (when a=v or b=v).
        # So we only need to check triples with v that do NOT contain u.
        for a in range(n):
            if a == v or a == u: continue
            for b in range(a+1, n):
                if b == v or b == u: continue
                common = adj[v] & adj[a] & adj[b]
                if common.bit_count() >= 3:
                    return False
        return True
