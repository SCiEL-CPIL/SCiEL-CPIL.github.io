"""Fresh design: Clebsch baseline + exact local search with fast bitset 3-col test."""
import time
import random

_N = 16
_ALL = (1 << _N) - 1
_EDGE_LIST = [(u, v) for u in range(_N) for v in range(u + 1, _N)]


def _clebsch_edges():
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (u ^ v) in (1, 2, 8):
                edges.append((u, v))
    return edges


def _build_adj(edge_set):
    adj = [0] * _N
    for (u, v) in edge_set:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def _is_3colorable(edge_set):
    adj = _build_adj(edge_set)
    n = _N
    colors = [-1] * n
    domains = [7] * n

    def search(uncolored):
        if uncolored == 0:
            return True
        # MRV
        best_v = -1
        best_c = 8
        for v in range(n):
            if (uncolored >> v) & 1:
                c = bin(domains[v]).count('1')
                if c < best_c:
                    best_c = c
                    best_v = v
                    if c <= 1:
                        break
        if best_v < 0:
            return True
        if best_c == 0:
            return False
        v = best_v
        new_uncolored = uncolored & ~(1 << v)
        d = domains[v]
        while d:
            lsb = d & (-d)
            d ^= lsb
            c = lsb.bit_length() - 1
            colors[v] = c
            neigh = adj[v] & new_uncolored
            changed = []
            ok = True
            tmp = neigh
            while tmp:
                lsb2 = tmp & (-tmp)
                tmp ^= lsb2
                w = lsb2.bit_length() - 1
                if (domains[w] >> c) & 1:
                    domains[w] &= ~(1 << c)
                    changed.append(w)
                    if domains[w] == 0:
                        ok = False
                        break
            if ok and search(new_uncolored):
                return True
            for w in changed:
                domains[w] |= (1 << c)
        colors[v] = -1
        return False

    return search(_ALL)


def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    base = set(_clebsch_edges())
    assert len(base) == 40
    # sanity: not 3-colorable
    if _is_3colorable(base):
        # fallback: return base anyway (should not happen)
        pass

    best = set(base)
    best_m = len(best)
    adj = _build_adj(best)

    def can_add(u, v, adj):
        return (adj[u] & adj[v]) == 0

    def local_search(edge_set, adj):
        nonlocal best, best_m
        improved = True
        while improved:
            improved = False
            # Adds
            for (u, v) in _EDGE_LIST:
                if (u, v) in edge_set:
                    continue
                if not can_add(u, v, adj):
                    continue
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                if not _is_3colorable(edge_set | {(u, v)}):
                    edge_set.add((u, v))
                    if len(edge_set) > best_m:
                        best = set(edge_set)
                        best_m = len(best)
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            # Swaps: remove one, then greedily add
            for (ru, rv) in list(edge_set):
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                edge_set.discard((ru, rv))
                # greedy add
                added = True
                while added:
                    added = False
                    for (u, v) in _EDGE_LIST:
                        if (u, v) in edge_set:
                            continue
                        if not can_add(u, v, adj):
                            continue
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        if not _is_3colorable(edge_set | {(u, v)}):
                            edge_set.add((u, v))
                            added = True
                            if len(edge_set) > best_m:
                                best = set(edge_set)
                                best_m = len(best)
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                # try re-add removed
                if can_add(ru, rv, adj):
                    adj[ru] |= 1 << rv
                    adj[rv] |= 1 << ru
                    if not _is_3colorable(edge_set | {(ru, rv)}):
                        edge_set.add((ru, rv))
                        if len(edge_set) > best_m:
                            best = set(edge_set)
                            best_m = len(best)
                    else:
                        adj[ru] &= ~(1 << rv)
                        adj[rv] &= ~(1 << ru)
                adj = _build_adj(edge_set)
                if len(edge_set) > best_m:
                    improved = True
        return edge_set, adj

    rng = random.Random(7)
    for restart in range(8):
        if time.time() > deadline:
            break
        if restart == 0:
            cur = set(best)
        else:
            cur = set(best)
            k = rng.randint(2, 6)
            for e in rng.sample(list(cur), min(k, len(cur))):
                cur.discard(e)
        adj = _build_adj(cur)
        cur, adj = local_search(cur, adj)
        if len(cur) > best_m:
            best = set(cur)
            best_m = len(best)

    return [list(e) for e in best]
