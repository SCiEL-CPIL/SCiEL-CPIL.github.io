import random
import time
import math

def construct():
    n = 16
    target_edges = 65
    deadline = time.time() + 140

    def compute_score(adj_bits, deg):
        max_sum = 0
        for i in range(n):
            ai = adj_bits[i]
            j = i + 1
            while j < n:
                if ai & (1 << j):
                    common = ai & adj_bits[j]
                    k = j + 1
                    while k < n:
                        if common & (1 << k):
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
                        k += 1
                j += 1
        return max_sum

    all_possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    best_overall_score = float('inf')
    best_overall_es = None

    max_restarts = 12

    for restart in range(max_restarts):
        if time.time() >= deadline - 5:
            break

        random.seed(1000 + restart * 7919)
        adj = [0] * n
        deg = [0] * n
        es = set()

        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                es.add((u, v))

        if restart % 2 == 0:
            a, b = random.sample(range(8), 2)
        else:
            a, b = random.sample(range(8, 16), 2)
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        deg[a] += 1
        deg[b] += 1
        es.add((a, b))

        score = compute_score(adj, deg)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_es = set(es)

        current_es = set(es)
        current_adj = adj[:]
        current_deg = deg[:]
        current_score = score

        temp = 2.0
        cool = 0.9997

        while time.time() < deadline - 5 and temp > 0.2:
            e1 = random.choice(tuple(current_es))
            u1, v1 = e1
            e2 = random.choice(all_possible_edges)
            if e2 in current_es:
                continue

            u2, v2 = e2

            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            current_deg[u1] -= 1
            current_deg[v1] -= 1
            current_es.discard(e1)

            current_adj[u2] |= (1 << v2)
            current_adj[v2] |= (1 << u2)
            current_deg[u2] += 1
            current_deg[v2] += 1
            current_es.add(e2)

            new_score = compute_score(current_adj, current_deg)

            delta = new_score - current_score
            if delta <= 0 or (delta < 50 and random.random() < math.exp(-delta / max(temp, 0.1))):
                current_score = new_score
                if new_score < best_overall_score:
                    best_overall_score = new_score
                    best_overall_es = set(current_es)
            else:
                current_adj[u2] &= ~(1 << v2)
                current_adj[v2] &= ~(1 << u2)
                current_deg[u2] -= 1
                current_deg[v2] -= 1
                current_es.discard(e2)

                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                current_deg[u1] += 1
                current_deg[v1] += 1
                current_es.add(e1)

            temp *= cool

    if best_overall_es is None:
        best_overall_es = set((u, v) for u in range(8) for v in range(8, 16))
        best_overall_es.add((0, 1))

    return {'n': n, 'edges': [[u, v] for u, v in best_overall_es]}
