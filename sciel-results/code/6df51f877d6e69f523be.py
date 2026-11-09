import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 16
    best_edges = []
    best_count = 0

    def greedy(seed):
        random.seed(seed)
        adj = [[False] * n for _ in range(n)]
        edges = []

        def would_create_q3(u, v):
            A = [w for w in range(n) if w != v and adj[u][w]]
            B = [w for w in range(n) if w != u and adj[v][w]]
            if len(A) < 2 or len(B) < 2:
                return False
            for a1, a2 in itertools.combinations(A, 2):
                for b1, b2 in itertools.combinations(B, 2):
                    if adj[a1][b1] and adj[a1][b2] and adj[a2][b1] and adj[a2][b2]:
                        return True
            return False

        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline - 5.0:
                break
            if not adj[u][v] and not would_create_q3(u, v):
                adj[u][v] = adj[v][u] = True
                edges.append([u, v])
        return edges

    for s in range(200):
        if time.time() > deadline - 10.0:
            break
        e = greedy(s)
        if len(e) > best_count:
            best_count = len(e)
            best_edges = e[:]

    return best_edges
