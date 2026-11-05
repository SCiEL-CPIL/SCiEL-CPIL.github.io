    def evaluate(scores):
        order = np.argsort(-scores).tolist()
        selected = np.zeros(n, dtype=bool)
        count = 0
        adj = adjacency
        sel = selected
        for v in order:
            if sel[v]:
                continue
            if np.any(sel[adj[v]]):
                continue
            sel[v] = True
            count += 1
        return count
