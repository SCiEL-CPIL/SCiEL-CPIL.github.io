            # Check girth
            girth = _get_girth(adj, n)
            if girth >= 10:
                # Valid
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
