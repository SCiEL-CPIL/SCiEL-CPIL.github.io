import time
import random
import numpy as np

def construct():
    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602),
    ]
    results = []
    total_start = time.time()
    total_limit = 140.0  # safety margin under 150s
    # Reserve time for each instance
    budget = [55.0, 80.0]
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start
        remaining = total_limit - elapsed
        if idx == len(instances) - 1:
            limit = remaining
        else:
            # Ensure the later instances get their reserved time
            reserved_later = sum(budget[idx + 1:])
            limit = min(budget[idx], max(1.0, remaining - reserved_later))
        if limit < 1.0:
            cliques = trivial_partition(n, p, seed)
        else:
            cliques = solve_instance(n, p, seed, limit)
        results.append(cliques)
    return results

def trivial_partition(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append([u, v])
    return edges

def generate_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def solve_instance(n, p, seed, time_limit):
    edges = generate_edges(n, p, seed)
    if not edges:
        return []

    # Bitset adjacency for residual graph
    res_adj = [0] * n
    for u, v in edges:
        res_adj[u] |= (1 << v)
        res_adj[v] |= (1 << u)

    cliques = []
    start_time = time.time()

    # ---- Phase 1: extract large cliques greedily ----
    extraction_budget = time_limit * 0.55
    rng = random.Random(12345)
    while time.time() - start_time < start_time + extraction_budget:
        # quick check: any edge left?
        any_edge = False
        for u in range(n):
            if res_adj[u]:
                any_edge = True
                break
        if not any_edge:
            break

        clique = find_large_clique(res_adj, n, rng)
        if not clique:
            # fallback: grab any single edge
            found = False
            for u in range(n):
                if res_adj[u]:
                    v = (res_adj[u] & -res_adj[u]).bit_length() - 1
                    clique = [u, v]
                    found = True
                    break
            if not found:
                break

        # remove all edges inside the clique from residual
        for i in range(len(clique)):
            u = clique[i]
            for j in range(i + 1, len(clique)):
                v = clique[j]
                res_adj[u] &= ~(1 << v)
                res_adj[v] &= ~(1 << u)
        cliques.append(clique)

    # ---- Phase 2: merge cliques that together form a larger clique ----
    merge_deadline = start_time + time_limit * 0.85
    changed = True
    while changed and time.time() < merge_deadline:
        changed = False
        best_gain = 0
        best_pair = None
        k = len(cliques)
        for i in range(k):
            c1 = cliques[i]
            if len(c1) < 2:
                continue
            c1_set = set(c1)
            # bitset of vertices in c1
            c1_mask = 0
            for x in c1:
                c1_mask |= (1 << x)
            for j in range(i + 1, k):
                c2 = cliques[j]
                if len(c2) < 2:
                    continue
                # merged set must be a clique: every vertex in c1 must connect to every vertex in c2
                # and within each is already a clique (by invariant)
                ok = True
                for u in c1:
                    # u must connect to all vertices in c2
                    missing = c1_mask  # not used; compute mask of c2
                    # check u against c2
                    for v in c2:
                        if u == v:
                            continue
                        if not (res_adj[u] & (1 << v)):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    # gain = number of cross edges we remove (all present)
                    gain = len(c1) * len(c2)
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, j)
        if best_pair is not None:
            i, j = best_pair
            c1 = cliques[i]
            c2 = cliques[j]
            # remove cross edges from residual
            for u in c1:
                for v in c2:
                    if u == v:
                        continue
                    res_adj[u] &= ~(1 << v)
                    res_adj[v] &= ~(1 << u)
            merged = list(set(c1) | set(c2))
            cliques[i] = merged
            cliques.pop(j)
            changed = True

    # ---- Phase 3: absorb remaining single edges into existing cliques ----
    absorb_deadline = start_time + time_limit * 0.97
    # collect remaining edges
    remaining = []
    for u in range(n):
        adj = res_adj[u]
        while adj:
            lb = adj & -adj
            v = lb.bit_length() - 1
            if u < v:
                remaining.append((u, v))
            adj ^= lb
    # try to fold each edge into a clique that already contains one endpoint
    for (u, v) in remaining:
        if time.time() > absorb_deadline:
            break
        if not (res_adj[u] & (1 << v)):
            continue  # already covered
        best_idx = -1
        best_size = 0
        for idx, clique in enumerate(cliques):
            if len(clique) < 2:
                continue
            if u in clique and v not in clique:
                # can we add v? v must connect to all in clique
                ok = True
                for w in clique:
                    if w != v and not (res_adj[v] & (1 << w)):
                        ok = False
                        break
                if ok and len(clique) > best_size:
                    best_size = len(clique)
                    best_idx = idx
            elif v in clique and u not in clique:
                ok = True
                for w in clique:
                    if w != u and not (res_adj[u] & (1 << w)):
                        ok = False
                        break
                if ok and len(clique) > best_size:
                    best_size = len(clique)
                    best_idx = idx
        if best_idx >= 0:
            clique = cliques[best_idx]
            for x in (u, v):
                if x not in clique:
                    clique.append(x)
                    for w in clique:
                        if w != x:
                            res_adj[x] &= ~(1 << w)
                            res_adj[w] &= ~(1 << x)
            # ensure the edge itself is removed (it will be, via the loop above when x connects to the other)
            res_adj[u] &= ~(1 << v)
            res_adj[v] &= ~(1 << u)

    # ---- Final: add any truly leftover edges as 2-cliques ----
    for u in range(n):
        adj = res_adj[u]
        while adj:
            lb = adj & -adj
            v = lb.bit_length() - 1
            if u < v:
                cliques.append([u, v])
            adj ^= lb

    return cliques

def find_large_clique(res_adj, n, rng):
    # degrees in residual
    deg = [res_adj[u].bit_count() for u in range(n)]
    # candidate pool: vertices with positive degree, sorted by degree desc with a bit of randomness
    verts = [u for u in range(n) if deg[u] > 0]
    if not verts:
        return []
    # top-K by degree, plus a few random
    verts.sort(key=lambda u: deg[u], reverse=True)
    topk = verts[:min(20, len(verts))]
    # add a couple random
    for _ in range(min(4, len(verts))):
        topk.append(rng.choice(verts))
    # dedupe preserving order
    seen = set()
    starts = []
    for u in topk:
        if u not in seen:
            seen.add(u)
            starts.append(u)

    best = []
    for s in starts:
        # greedy expansion
        C = [s]
        P = res_adj[s]
        while P:
            # pick vertex in P with max (residual degree within P)
            best_u = -1
            best_score = -1
            tmp = P
            while tmp:
                lb = tmp & -tmp
                u = lb.bit_length() - 1
                score = (P & res_adj[u]).bit_count()
                if score > best_score:
                    best_score = score
                    best_u = u
                tmp ^= lb
            if best_u < 0:
                break
            C.append(best_u)
            P &= res_adj[best_u]
        if len(C) > len(best):
            best = C
        # also try a small branch-and-bound from s to possibly beat greedy
        # limit the search: only consider a small candidate set (top by degree in P)
        cand = sorted([u for u in range(n) if (res_adj[s] & (1 << u))],
                      key=lambda u: (res_adj[s] & res_adj[u]).bit_count(),
                      reverse=True)[:12]
        # run a bounded BK
        local_best = _bk_expand(set([s]), frozenset(cand), res_adj, n, len(C), deadline=None)
        if len(local_best) > len(best):
            best = local_best
    return best

def _bk_expand(C, P, res_adj, n, best_len, deadline):
    # Bron–Kerbosch with pivot, bounded by current best length
    # C: current clique (set), P: candidates (frozenset)
    if not P:
        return list(C)
    # if even adding all of P can't beat best_len, prune
    if len(C) + len(P) <= best_len:
        return list(C)
    # pivot: choose u in P ∪ X with max |P ∩ N(u)|; here X empty, so pick u in P
    # pick pivot with max degree in P
    pivot = max(P, key=lambda u: len(P & _neighbors(u, res_adj)))
    # candidates: P \ N(pivot)
    P_not_N = P - _neighbors(pivot, res_adj)
    best = list(C)
    for v in list(P_not_N):
        new_C = set(C)
        new_C.add(v)
        new_P = P & _neighbors(v, res_adj)
        sub = _bk_expand(new_C, frozenset(new_P), res_adj, n, len(best), deadline)
        if len(sub) > len(best):
            best = sub
        P.discard(v)
    return best

def _neighbors(u, res_adj):
    # return a set of neighbors from bitset
    s = set()
    adj = res_adj[u]
    while adj:
        lb = adj & -adj
        v = lb.bit_length() - 1
        s.add(v)
        adj ^= lb
    return s
