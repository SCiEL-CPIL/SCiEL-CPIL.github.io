        import random
        r = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if r.random() < p:
                    edges.append((u, v))
        return edges
