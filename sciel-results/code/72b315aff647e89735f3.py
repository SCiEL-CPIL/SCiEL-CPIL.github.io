        # Pick edge to remove
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and (adj[u] & (1 << v)):
                break
