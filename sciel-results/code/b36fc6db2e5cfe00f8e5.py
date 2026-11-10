import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 20
    best_edges = []
    best_count = 0
    rng = random.Random(42)

    def cn(adj, u, v):
        return int(np.dot(adj[u], adj[v]))

    def repair(adj):
        for _ in range(300):
            if time.time() - start_time > deadline - 8:
                break
            ch = False
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v] and cn(adj, u, v) >= 3:
                        adj[u, v] = 0
                        adj[v, u] = 0
                        ch = True
            if not ch:
                break
        return adj

    def can_add(adj, u, v):
        if adj[u, v]:
            return True
        c = cn(adj, u, v)
        if c >= 3:
            return False
        for w in range(n):
            if w != u and w != v and adj[u, w] and adj[v, w]:
                if cn(adj, u, w) >= 2 or cn(adj, v, w) >= 2:
                    return False
        return True

    def greedy_add(adj):
        imp = True
        while imp and time.time() - start_time < deadline - 8:
            imp = False
            cands = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u, v]:
                        c = cn(adj, u, v)
                        if c < 3:
                            cands.append((c, u, v))
            cands.sort()
            for _, u, v in cands:
                if can_add(adj, u, v):
                    adj[u, v] = 1
                    adj[v, u] = 1
                    imp = True
        return adj

    def local_search(adj, iters=4000):
        for _ in range(iters):
            if time.time() - start_time > deadline - 8:
                break
            added = False
            for _ in range(50):
                u, v = rng.sample(range(n), 2)
                if not adj[u, v] and can_add(adj, u, v):
                    adj[u, v] = 1
                    adj[v, u] = 1
                    added = True
                    break
            if not added:
                elist = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                if elist:
                    a, b = rng.choice(elist)
                    for _ in range(20):
                        c, d = rng.sample(range(n), 2)
                        if adj[c, d] or {c, d} == {a, b}:
                            continue
                        adj[a, b] = 0
                        adj[b, a] = 0
                        if can_add(adj, c, d):
                            adj[c, d] = 1
                            adj[d, c] = 1
                            break
                        else:
                            adj[a, b] = 1
                            adj[b, a] = 1
        return adj

    for restart in range(300):
        if time.time() - start_time > deadline - 10:
            break
        adj = np.zeros((n, n), dtype=np.int8)
        r = restart % 4
        if r == 0:
            dens = 0.3 + 0.4 * rng.random()
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < dens:
                        adj[u, v] = 1
                        adj[v, u] = 1
        elif r == 1:
            for u in range(10):
                for v in range(10, 20):
                    if rng.random() < 0.55:
                        adj[u, v] = 1
                        adj[v, u] = 1
            for u in range(10):
                for v in range(u + 1, 10):
                    if rng.random() < 0.45:
                        adj[u, v] = 1
                        adj[v, u] = 1
            for u in range(10, 20):
                for v in range(u + 1, 20):
                    if rng.random() < 0.45:
                        adj[u, v] = 1
                        adj[v, u] = 1
        elif r == 2:
            for u in range(10):
                for v in range(10, 20):
                    adj[u, v] = 1
                    adj[v, u] = 1
            for u in range(10):
                for v in range(u + 1, 10):
                    if rng.random() < 0.3:
                        adj[u, v] = 1
                        adj[v, u] = 1
            for u in range(10, 20):
                for v in range(u + 1, 20):
                    if rng.random() < 0.3:
                        adj[u, v] = 1
                        adj[v, u] = 1
        else:
            # Start from best and perturb
            if best_edges:
                for u, v in best_edges:
                    adj[u, v] = 1
                    adj[v, u] = 1
            # Randomly remove some edges
            elist = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            for _ in range(rng.randint(0, 10)):
                if elist:
                    a, b = rng.choice(elist)
                    adj[a, b] = 0
                    adj[b, a] = 0
                    elist.remove((a, b))

        adj = repair(adj)
        adj = greedy_add(adj)
        adj = local_search(adj, iters=2000)

        cnt = int(np.sum(adj)) // 2
        if cnt > best_count:
            best_count = cnt
            best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    return best_edges
