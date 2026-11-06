import time
import random

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 115.0
    N = 40
    n_pairs = N * (N - 1) // 2
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    def count_c6(adj, adj_set):
        """Count C6s. Each C6 contributes 6 to sum of paths3 over non-adjacent pairs."""
        total = 0
        for u in range(N):
            nu = adj[u]
            for v in range(u + 1, N):
                if (u, v) in adj_set:
                    continue
                nv = adj[v]
                p3 = 0
                for a in nu:
                    na = adj[a]
                    for b in nv:
                        if b != a and (a, b) in adj_set:
                            p3 += 1
                total += p3
        return total // 6

    def greedy_build():
        adj = [set() for _ in range(N)]
        adj_set = set()
        random.shuffle(all_pairs)
        for (i, j) in all_pairs:
            adj[i].add(j)
            adj[j].add(i)
            adj_set.add((i, j))
            if count_c6(adj, adj_set) > 0:
                adj[i].remove(j)
                adj[j].remove(i)
                adj_set.discard((i, j))
        return adj, adj_set

    def local_search(adj, adj_set, tlimit):
        best_e = len(adj_set)
        best_adj = [set(s) for s in adj]
        best_set = set(adj_set)
        stall = 0
        while time.time() < tlimit and stall < 300:
            non_edges = [(i, j) for (i, j) in all_pairs if (i, j) not in adj_set]
            if not non_edges:
                break
            random.shuffle(non_edges)
            found = False
            for (i, j) in non_edges:
                if time.time() >= tlimit:
                    break
                adj[i].add(j); adj[j].add(i); adj_set.add((i, j))
                if count_c6(adj, adj_set) == 0:
                    best_e = len(adj_set)
                    best_adj = [set(s) for s in adj]
                    best_set = set(adj_set)
                    stall = 0
                    found = True
                    break
                adj[i].remove(j); adj[j].remove(i); adj_set.discard((i, j))
            if not found:
                stall += 1
                # Try 1-out-2-in swap
                if random.random() < 0.6:
                    el = list(adj_set)
                    random.shuffle(el)
                    for (a, b) in el:
                        if time.time() >= tlimit:
                            break
                        adj[a].remove(b); adj[b].remove(a); adj_set.discard((a, b))
                        ne = [(i, j) for (i, j) in all_pairs if (i, j) not in adj_set]
                        random.shuffle(ne)
                        added = []
                        for (i, j) in ne:
                            adj[i].add(j); adj[j].add(i); adj_set.add((i, j))
                            added.append((i, j))
                            if count_c6(adj, adj_set) > 0:
                                adj[i].remove(j); adj[j].remove(i); adj_set.discard((i, j))
                                added.pop()
                                break
                        if len(added) == 2:
                            best_e = len(adj_set)
                            best_adj = [set(s) for s in adj]
                            best_set = set(adj_set)
                            stall = 0
                            break
                        for (i, j) in added:
                            adj[i].remove(j); adj[j].remove(i); adj_set.discard((i, j))
                        adj[a].add(b); adj[b].add(a); adj_set.add((a, b))
        return best_adj, best_set

    adj, adj_set = greedy_build()
    adj, adj_set = local_search(adj, adj_set, deadline)
    best_e = len(adj_set)
    best_adj = [set(s) for s in adj]
    best_set = set(adj_set)

    while time.time() < deadline - 10:
        a2, s2 = greedy_build()
        a2, s2 = local_search(a2, s2, deadline)
        if len(s2) > best_e:
            best_e = len(s2)
            best_adj = [set(s) for s in a2]
            best_set = set(s2)

    edges = sorted([i, j] for (i, j) in best_set)
    return edges
# EVOLVE-BLOCK-END
