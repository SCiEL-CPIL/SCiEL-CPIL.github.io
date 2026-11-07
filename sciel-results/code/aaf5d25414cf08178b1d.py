        # Pre-sort adjacency lists by degree
        for i in range(k):
            adj[i].sort(key=lambda x: len(adj[x]), reverse=True)
