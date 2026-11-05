        # Initialize best with known feasible K_{2,18}
        best_adj = np.zeros((n, n), dtype=np.uint8)
        for a in range(2):
            for b in range(2, n):
                best_adj[a, b] = 1
                best_adj[b, a] = 1
        best_edges = 36
        best_penalty = 0
