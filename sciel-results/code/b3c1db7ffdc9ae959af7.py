import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin before 150s kill

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_pairs_set = set(all_pairs)

    def adj_to_edges(adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i] >> j & 1]

    def common_count(adj, a, b):
        return (adj[a] & adj[b]).bit_count()

    def can_add(adj, a, b):
        # adding edge a-b creates K3,3 iff some c (c!=a,b) has >=3 common neighbors with a and b
        # common neighbors of a and b after adding: (adj[a] & adj[b]) plus each other
        mask = (adj[a] & adj[b]) | (1 << a) | (1 << b)
        for c in range(n):
            if c == a or c == b:
                continue
            if common_count(adj, a, c) >= 3 and common_count(adj, b, c) >= 3:
                # c shares >=3 with a and >=3 with b; need intersection of the shared sets size>=3
                # shared between a and c: adj[a]&adj[c]; between b and c: adj[b]&adj[c]
                # a K3,3 with parts {a,b,c} and 3 common neighbors exists iff
                # |adj[a]&adj[b]&adj[c]| >= 3
                if (adj[a] & adj[b] & adj[c]).bit_count() >= 3:
                    return False
        return True

    def is_valid(adj):
        # full check: no triple has >=3 common neighbors
        for i in range(n):
            for j in range(i + 1, n):
                common_ij = adj[i] & adj[j]
                # for any k, if common_ij & adj[k] has >=3 then K3,3
                for k in range(j + 1, n):
                    if (common_ij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def greedy_build(order):
        adj = [0] * n
        for (a, b) in order:
            if can_add(adj, a, b):
                adj[a] |= 1 << b
                adj[b] |= 1 << a
        return adj

    def anneal(adj, steps, start_temp, end_temp):
        cur_edges = count_edges(adj)
        best_adj = list(adj)
        best_edges = cur_edges
        for s in range(steps):
            if time.time() > deadline:
                break
            temp = start_temp * (end_temp / start_temp) ** (s / max(1, steps))
            a = random.randrange(n)
            b = random.randrange(n)
            if a == b or (adj[a] >> b) & 1:
                continue
            if can_add(adj, a, b):
                adj[a] |= 1 << b
                adj[b] |= 1 << a
                cur_edges += 1
                if cur_edges > best_edges:
                    best_edges = cur_edges
                    best_adj = list(adj)
            else:
                # try removing a random existing edge to allow re-adding
                ea = random.randrange(n)
                eb = random.randrange(n)
                if ea == eb or not ((adj[ea] >> eb) & 1):
                    continue
                adj[ea] &= ~(1 << eb)
                adj[eb] &= ~(1 << ea)
                cur_edges -= 1
                # now try to add a-b
                if can_add(adj, a, b):
                    adj[a] |= 1 << b
                    adj[b] |= 1 << a
                    cur_edges += 1
                else:
                    # restore
                    adj[ea] |= 1 << eb
                    adj[eb] |= 1 << ea
                    cur_edges += 1
                if cur_edges > best_edges:
                    best_edges = cur_edges
                    best_adj = list(adj)
        return best_adj, best_edges

    # Phase 1: several greedy builds with different random orderings
    best_adj = None
    best_edges = -1
    random.seed(12345)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = all_pairs[:]
        random.shuffle(order)
        adj = greedy_build(order)
        e = count_edges(adj)
        if e > best_edges:
            best_edges = e
            best_adj = list(adj)

    # Phase 2: simulated annealing from the best
    if best_adj is not None and time.time() < deadline:
        best_adj, best_edges = anneal(best_adj, 200000, 1.0, 0.05)

    # Phase 3: local improvement - try adding any missing edge
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for (a, b) in all_pairs:
                if (best_adj[a] >> b) & 1:
                    continue
                if can_add(best_adj, a, b):
                    best_adj[a] |= 1 << b
                    best_adj[b] |= 1 << a
                    best_edges += 1
                    improved = True

    # Final safety: verify validity
    if not is_valid(best_adj):
        # fall back to safe K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]

    return adj_to_edges(best_adj)
# EVOLVE-BLOCK-END
