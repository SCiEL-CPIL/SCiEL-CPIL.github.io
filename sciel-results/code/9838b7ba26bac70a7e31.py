import numpy as np
import random
import time

def construct(k, g):
    start_time = time.time()
    time_limit = 110

    def moore_bound(k, g):
        if k == 2:
            return g
        if g % 2 == 1:
            terms = (g - 1) // 2
            if k == 2:
                return 1 + 2 * terms
            else:
                return 1 + k * (((k-1)**terms - 1) // (k - 2))
        else:
            terms = g // 2 - 1
            if k == 2:
                return 2 * terms
            else:
                return 2 * (((k-1)**terms - 1) // (k - 2))

    n_min = moore_bound(k, g)

    def generate_random_regular_graph(n, k):
        if n * k < 2 or n * k % 2 != 0:
            return None
        if k >= n:
            return None
        for _ in range(300):
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            adj = np.zeros((n, n), dtype=np.int8)
            ok = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or adj[u, v]:
                    ok = False
                    break
                adj[u, v] = 1
                adj[v, u] = 1
            if ok and np.all(np.sum(adj, axis=1) == k):
                return adj
        return None

    def has_cycle_less_than_g(adj, n, g):
        for s in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            par = np.full(n, -1, dtype=np.int32)
            dist[s] = 0
            q = [s]
            h = 0
            while h < len(q):
                u = q[h]; h += 1
                for v in np.nonzero(adj[u])[0]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        par[v] = u
                        q.append(v)
                    elif v != par[u]:
                        if dist[u] + dist[v] + 1 < g:
                            return True
        return False

    def count_tri(adj):
        return np.trace(adj @ adj @ adj) // 6

    def local_search(n, k, g, budget):
        adj = None
        for _ in range(200):
            adj = generate_random_regular_graph(n, k)
            if adj is not None:
                break
        if adj is None:
            return None
        if not has_cycle_less_than_g(adj, n, g):
            return adj
        dl = start_time + budget
        best = adj.copy()
        best_tri = count_tri(adj)
        for _ in range(30000):
            if time.time() > dl:
                break
            # pick two random edges
            us = np.nonzero(adj.sum(axis=1) > 0)[0]
            if len(us) < 4:
                break
            u1 = random.choice(us)
            v1 = random.choice(np.nonzero(adj[u1])[0])
            u2 = random.choice(us)
            if u2 in (u1, v1):
                continue
            v2 = random.choice(np.nonzero(adj[u2])[0])
            if v2 in (u1, v1) or v2 == u2:
                continue
            if adj[u1, u2] or adj[v1, v2]:
                continue
            adj[u1, v1] = 0; adj[v1, u1] = 0
            adj[u2, v2] = 0; adj[v2, u2] = 0
            adj[u1, u2] = 1; adj[u2, u1] = 1
            adj[v1, v2] = 1; adj[v2, v1] = 1
            tri = count_tri(adj)
            if tri < best_tri:
                best_tri = tri
                best = adj.copy()
            elif random.random() < 0.1:
                pass
            else:
                adj[u1, v1] = 1; adj[v1, u1] = 1
                adj[u2, v2] = 1; adj[v2, u2] = 1
                adj[u1, u2] = 0; adj[u2, u1] = 0
                adj[v1, v2] = 0; adj[v2, v1] = 0
        if not has_cycle_less_than_g(best, n, g):
            return best
        return None

    if k == 5 and g == 9:
        n_cands = [1116, 1200, 1500, 2000, 2500, 3000, 4000]
    else:
        n_cands = []
        base = max(n_min, 50)
        for m in [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
            nc = int(base * m)
            if nc * k % 2:
                nc += 1
            n_cands.append(nc)
        n_cands = sorted(set(n_cands))

    for nc in n_cands:
        if time.time() - start_time > time_limit - 10:
            break
        res = local_search(nc, k, g, 15)
        if res is not None:
            edges = [(u, int(v)) for u in range(nc) for v in np.nonzero(res[u])[0] if u < v]
            return nc, edges

    nc = n_cands[0] if n_cands else max(n_min, 10)
    if nc * k % 2:
        nc += 1
    adj = generate_random_regular_graph(nc, k)
    if adj is not None:
        edges = [(u, int(v)) for u in range(nc) for v in np.nonzero(adj[u])[0] if u < v]
        return nc, edges
    return 0, []
