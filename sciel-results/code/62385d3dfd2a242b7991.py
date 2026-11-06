import time
import sys
from random import Random


def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s; keep explicit deadline guard.
    deadline = start_time + 138.0

    def gen_graph(n, p, seed):
        rng = Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def maximal_cliques(adj, n):
        # Bron-Kerbosch with pivot; yields maximal cliques as frozensets
        # of vertex ids.
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    yield frozenset(R)
                return
            # pivot selection
            if P:
                # choose pivot u in P ∪ X with max |P ∩ N(u)|
                union = P | X
                best_u = None
                best_cnt = -1
                for u in union:
                    cnt = len(P & adj[u])
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_u = u
                candidates = P - adj[best_u]
            else:
                candidates = P
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                P.discard(v)
                X.add(v)
                for c in bk(new_R, new_P, new_X):
                    yield c
                # undo
                X.discard(v)
                P.add(v)

        for c in bk(set(), set(range(n)), set()):
            yield c

    def local_search(chosen, cliques_by_v, all_cliques, rng):
        # chosen: set of vertices. cliques_by_v: v -> frozenset of cliques containing v
        # all_cliques: list of cliques (frozensets)
        chosen = set(chosen)
        # coverage count per clique
        cov = {}
        for c in all_cliques:
            cnt = 0
            for v in c:
                if v in chosen:
                    cnt += 1
            cov[c] = cnt

        def is_valid():
            for c in all_cliques:
                if cov[c] < 1:
                    return False
            return True

        improved = True
        while improved:
            improved = False
            # removal pass
            for v in list(chosen):
                if len(chosen) <= 1:
                    continue
                # count cliques that would become uncovered if we remove v
                lost = 0
                for c in cliques_by_v[v]:
                    if cov[c] == 1:
                        lost += 1
                if lost == 0:
                    # safe to remove
                    chosen.discard(v)
                    for c in cliques_by_v[v]:
                        cov[c] -= 1
                    improved = True
            if not is_valid():
                # shouldn't happen; but guard
                break
            # swap pass
            # For each chosen v, consider removing v and adding one non-chosen w
            # to restore coverage of cliques that become uncovered.
            for v in list(chosen):
                if len(chosen) <= 1:
                    continue
                # cliques that become uncovered
                uncovered = []
                for c in cliques_by_v[v]:
                    if cov[c] == 1:
                        uncovered.append(c)
                if not uncovered:
                    continue
                # find a w not in chosen that covers all of uncovered (or reduces most)
                # candidate w: in union of uncovered cliques
                cand = set()
                for c in uncovered:
                    cand |= c
                cand -= chosen
                best_w = None
                best_gain = -1
                for w in cand:
                    gain = 0
                    for c in uncovered:
                        if w in c:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_w = w
                if best_w is None:
                    continue
                # only swap if it keeps validity (covers all uncovered) and size not larger
                if best_gain >= len(uncovered):
                    # perform swap
                    chosen.discard(v)
                    for c in cliques_by_v[v]:
                        cov[c] -= 1
                    chosen.add(best_w)
                    for c in cliques_by_v[best_w]:
                        cov[c] += 1
                    improved = True
        return chosen

    def solve_instance(n, p, seed, budget):
        adj = gen_graph(n, p, seed)
        cliques = list(maximal_cliques(adj, n))
        if not cliques:
            return []

        # map vertex -> set of cliques containing it
        cliques_by_v = [set() for _ in range(n)]
        for idx, c in enumerate(cliques):
            for v in c:
                cliques_by_v[v].add(idx)

        # convert to index-based for speed
        cliques_idx = [set(c) for c in cliques]

        def greedy_set_cover(rng):
            # returns set of chosen vertex indices
            remaining = set(range(len(cliques_idx)))
            chosen = set()
            while remaining:
                best_v = None
                best_cnt = -1
                for v in range(n):
                    if v in chosen:
                        continue
                    cnt = 0
                    for ci in remaining:
                        if v in cliques_idx[ci]:
                            cnt += 1
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_v = v
                if best_v is None or best_cnt <= 0:
                    break
                chosen.add(best_v)
                new_remaining = set()
                for ci in remaining:
                    if best_v not in cliques_idx[ci]:
                        new_remaining.add(ci)
                remaining = new_remaining
            return chosen

        def local_search_idx(chosen, rng):
            chosen = set(chosen)
            cov = [0] * len(cliques_idx)
            for v in chosen:
                for ci in cliques_by_v[v]:
                    cov[ci] += 1

            improved = True
            while improved:
                improved = False
                for v in list(chosen):
                    if len(chosen) <= 1:
                        continue
                    lost = 0
                    for ci in cliques_by_v[v]:
                        if cov[ci] == 1:
                            lost += 1
                    if lost == 0:
                        chosen.discard(v)
                        for ci in cliques_by_v[v]:
                            cov[ci] -= 1
                        improved = True
                for v in list(chosen):
                    if len(chosen) <= 1:
                        continue
                    uncovered = []
                    for ci in cliques_by_v[v]:
                        if cov[ci] == 1:
                            uncovered.append(ci)
                    if not uncovered:
                        continue
                    cand = set()
                    for ci in uncovered:
                        cand |= cliques_idx[ci]
                    cand -= chosen
                    best_w = None
                    best_gain = -1
                    for w in cand:
                        gain = 0
                        for ci in uncovered:
                            if w in cliques_idx[ci]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_w = w
                    if best_w is None:
                        continue
                    if best_gain >= len(uncovered):
                        chosen.discard(v)
                        for ci in cliques_by_v[v]:
                            cov[ci] -= 1
                        chosen.add(best_w)
                        for ci in cliques_by_v[best_w]:
                            cov[ci] += 1
                        improved = True
            return chosen

        def valid(chosen):
            cov = [0] * len(cliques_idx)
            for v in chosen:
                for ci in cliques_by_v[v]:
                    cov[ci] += 1
            for ci in range(len(cliques_idx)):
                if cov[ci] < 1:
                    return False
            return True

        best = None
        best_size = float('inf')
        rng = Random(seed * 1000 + 7)

        # initial greedy
        g = greedy_set_cover(rng)
        g = local_search_idx(g, rng)
        if valid(g):
            best = set(g)
            best_size = len(g)

        # randomized restarts within budget
        attempts = 0
        while time.time() < budget and best_size > 0:
            attempts += 1
            # randomized greedy: break ties randomly
            remaining = set(range(len(cliques_idx)))
            chosen = set()
            while remaining:
                best_v = None
                best_cnt = -1
                for v in range(n):
                    if v in chosen:
                        continue
                    cnt = 0
                    for ci in remaining:
                        if v in cliques_idx[ci]:
                            cnt += 1
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_v = v
                    elif cnt == best_cnt and cnt > 0:
                        # random tie break
                        if rng.random() < 0.5:
                            best_v = v
                if best_v is None or best_cnt <= 0:
                    break
                chosen.add(best_v)
                new_remaining = set()
                for ci in remaining:
                    if best_v not in cliques_idx[ci]:
                        new_remaining.add(ci)
                remaining = new_remaining
            chosen = local_search_idx(chosen, rng)
            if valid(chosen) and len(chosen) < best_size:
                best = set(chosen)
                best_size = len(chosen)
            if best_size == 1:
                break

        if best is None:
            # fallback: greedy result
            best = set(g)
        return sorted(best)

    out = []
    # Allocate budget across instances; keep safety margin.
    total_budget = deadline - time.time()
    # Give a bit more to larger instances.
    weights = [1.0, 1.2, 1.5]
    wsum = sum(weights)
    budgets = [max(5.0, total_budget * w / wsum) for w in weights]

    for (n, p, seed), b in zip(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)), budgets):
        if time.time() > deadline - 2.0:
            # not enough time; return best-effort empty? Better to do quick greedy.
            b = max(1.0, deadline - time.time() - 1.0)
        cover = solve_instance(n, p, seed, b)
        out.append(cover)

    return out
