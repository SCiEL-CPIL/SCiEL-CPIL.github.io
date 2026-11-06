import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = 143

    def elapsed():
        return time.time() - start_time

    def c4_violations(adj, n):
        """Return list of (i, j, w, x) forming C4 i-w-j-x-i (or equivalent)."""
        violations = []
        for i in range(n):
            adj_i = adj[i]
            for j in range(i + 1, n):
                common = adj_i & adj[j]
                if common.bit_count() >= 2:
                    # Get two common neighbors
                    tmp = common
                    c1 = None
                    c2 = None
                    while tmp:
                        lsb = tmp & -tmp
                        idx = lsb.bit_length() - 1
                        if c1 is None:
                            c1 = idx
                        else:
                            c2 = idx
                            break
                        tmp ^= lsb
                    if c1 is not None and c2 is not None:
                        violations.append((i, j, c1, c2))
        return violations

    def is_c4_free(adj, n):
        for i in range(n):
            adj_i = adj[i]
            for j in range(i + 1, n):
                if (adj_i & adj[j]).bit_count() >= 2:
                    return False
        return True

    def min_deg(adj, n):
        m = n
        for i in range(n):
            d = adj[i].bit_count()
            if d < m:
                m = d
        return m

    def build_edges(adj, n):
        edges = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    edges.append([i, j])
        return edges

    def try_n(n, budget):
        target = n - 39
        if target < 1:
            target = 1
        best = None
        best_md = -1

        # Try several density starts
        ps = [target / n, (target + 1) / n, (target + 2) / n, 0.55]
        ps = [p for p in ps if 0 < p < 0.9]
        if not ps:
            ps = [0.5]

        for p in ps:
            for attempt in range(4):
                if elapsed() > budget:
                    break
                # Random graph
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)

                # Repair: iteratively remove one edge per detected C4
                max_iter = 200
                for it in range(max_iter):
                    if elapsed() > budget:
                        break
                    v = c4_violations(adj, n)
                    if not v:
                        break
                    # Remove a random edge from a random C4
                    i, j, c1, c2 = random.choice(v)
                    # Remove edge (i, c1)
                    adj[i] &= ~(1 << c1)
                    adj[c1] &= ~(1 << i)

                if not is_c4_free(adj, n):
                    continue

                md = min_deg(adj, n)
                if md > best_md:
                    best_md = md
                    best = adj[:]

                # Try to raise minimum degree via edge additions
                degs = [adj[i].bit_count() for i in range(n)]
                for _ in range(60):
                    if elapsed() > budget:
                        break
                    if min(degs) >= target:
                        break
                    # Find a low-degree vertex
                    low = [i for i in range(n) if degs[i] <= min(degs) + 1]
                    random.shuffle(low)
                    progressed = False
                    for u in low:
                        if degs[u] >= target:
                            continue
                        non = (~adj[u]) & ((1 << n) - 1)
                        non &= ~(1 << u)
                        if not non:
                            continue
                        tmp = non
                        cands = []
                        while tmp:
                            lsb = tmp & -tmp
                            idx = lsb.bit_length() - 1
                            cands.append(idx)
                            tmp ^= lsb
                        random.shuffle(cands)
                        for v in cands:
                            if (adj[u] & adj[v]).bit_count() < 2:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                degs[u] += 1
                                degs[v] += 1
                                progressed = True
                                if degs[u] >= target:
                                    break
                        if degs[u] >= target:
                            continue
                    if not progressed:
                        break

                if is_c4_free(adj, n):
                    md = min_deg(adj, n)
                    if md > best_md:
                        best_md = md
                        best = adj[:]
                    if md >= target:
                        return n, build_edges(adj, n)

        if best is not None and best_md >= target:
            return n, build_edges(best, n)
        return None

    # Allocate budget per n: prefer higher n
    total_budget = deadline
    ns = [47, 46, 45, 44, 43, 42, 41, 40]
    # Give more time to higher n
    weights = [1.5, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
    total_w = sum(weights)
    alloc = [int(total_budget * w / total_w) for w in weights]
    # Ensure minimum 5s each
    alloc = [max(5, a) for a in alloc]
    # Scale down if sum exceeds budget
    s = sum(alloc)
    if s > total_budget:
        scale = total_budget / s
        alloc = [max(3, int(a * scale)) for a in alloc]

    for n, b in zip(ns, alloc):
        if elapsed() > deadline - 1:
            break
        res = try_n(n, deadline - 1)
        if res is not None:
            return {'n': res[0], 'edges': res[1]}

    # Fallback
    return {'n': 39, 'edges': []}
