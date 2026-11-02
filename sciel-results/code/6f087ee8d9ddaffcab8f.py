    def generate_random_k_regular(n, k):
        # Ensure n*k is even
        if (n * k) % 2 != 0: return None
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None # Retry
            if v in adj[u]:
                return None # Retry
            adj[u].add(v)
            adj[v].add(u)
        return adj
    