    def improve(adj):
        # Try to add edges
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # Iterate non-edges
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj[i]:
                        # Try adding (i, j)
                        adj[i].add(j)
                        adj[j].add(i)
                        
                        # Check validity
                        # Only triples involving i or j can be affected
                        valid = True
                        
                        # 1. Check triples {i, a, b} where a, b != i, j
                        for a in range(n):
                            if a == i or a == j: continue
                            for b in range(a+1, n):
                                if b == i or b == j: continue
                                common = adj[i] & adj[a] & adj[b]
                                if len(common) >= 3:
                                    valid = False
                                    break
                            if not valid: break
                        
                        # 2. Check triples {i, j, b} where b != i, j
                        if valid:
                            for b in range(n):
                                if b == i or b == j: continue
                                common = adj[i] & adj[j] & adj[b]
                                if len(common) >= 3:
                                    valid = False
                                    break
                        
                        # 3. Check triples {j, a, b} where a, b != i, j
                        # This was missing!
                        if valid:
                            for a in range(n):
                                if a == i or a == j: continue
                                for b in range(a+1, n):
                                    if b == i or b == j: continue
                                    common = adj[j] & adj[a] & adj[b]
                                    if len(common) >= 3:
                                        valid = False
                                        break
                                if not valid: break
                        
                        if valid:
                            improved = True
                        else:
                            adj[i].remove(j)
                            adj[j].remove(i)
