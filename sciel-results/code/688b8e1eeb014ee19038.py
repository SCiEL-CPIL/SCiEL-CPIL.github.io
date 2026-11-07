    def is_clique(verts):
        for u, v in combinations(verts, 2):
            if v not in adj[u]:
                return False
        return True
