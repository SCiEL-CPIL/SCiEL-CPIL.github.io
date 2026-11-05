    def get_score_fast(adj):
        max_sum = 0
        # Precompute degrees
        degrees = np.sum(adj, axis=1)
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    # Common neighbors
                    # adj[u] is a vector of 0/1
                    # common = adj[u] & adj[v]
                    common = np.where(np.logical_and(adj[u], adj[v]))[0]
                    for w in common:
                        if w != u and w != v:
                            s = degrees[u] + degrees[v] + degrees[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum
