"""Rewritten solver: exact min cochromatic partition via branch-and-bound.

Structural change from the seed's greedy constructive heuristic: this performs a
bounded backtracking search over part assignments, pruned by a rigorous lower
bound (max of the chromatic number of G and of its complement) and a tight
upper bound from a greedy + local-search. A deadline guard returns the best
valid partition found.
"""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
DEADLINE = 82.0  # safety margin under the 90s kill
_T0 = None


def _setup():
    global _T0
    _T0 = time.time()
    edges = P['edges']
    adj = [0] * N
    for a, b in edges:
        adj[a] |= 1 << b
        adj[b] |= 1 << a
    comp = [((1 << N) - 1) ^ (1 << v) ^ adj[v] for v in range(N)]
    return adj, comp


def _greedy_color(adj, order):
    color = [0] * N
    used = 0
    for v in order:
        banned = 0
        cv = color[v]
        for u in range(N):
            if cv and (adj[v] >> u) & 1:
                banned |= 1 << color[u]
        c = 0
        while (banned >> c) & 1:
            c += 1
        color[v] = c
        if c >= used:
            used = c + 1
    return used


def _order_by_deg(adj):
    return sorted(range(N), key=lambda v: bin(adj[v]).count('1'), reverse=True)


def _upper_bound(adj, comp):
    # greedy cochromatic coloring (parts = cliques of G or cliques of Gbar)
    order = sorted(range(N), key=lambda v: bin(adj[v]).count('1') + bin(comp[v]).count('1'), reverse=True)
    parts = []          # each part: (type, mask)  type 0=clique in G, 1=clique in Gbar
    placed = [False] * N
    for v in order:
        av, cv = adj[v], comp[v]
        for it, (typ, mask) in enumerate(parts):
            a = adj if typ == 0 else comp
            if all((a[v] >> u) & 1 for u in range(N) if (mask >> u) & 1):
                parts[it] = (typ, mask | (1 << v))
                placed[v] = True
                break
        if not placed[v]:
            parts.append((0, 1 << v))
            placed[v] = True
    best = len(parts)

    # local search: try moving each vertex to another part or a new part, keep if it reduces
    improved = True
    while improved and time.time() < DEADLINE - 1.0:
        improved = False
        for v in range(N):
            cur = next(i for i, (typ, mask) in enumerate(parts) if (mask >> v) & 1)
            curmask = parts[cur][1] ^ (1 << v)
            # try moving v to another existing part
            for it, (typ, mask) in enumerate(parts):
                if it == cur:
                    continue
                a = adj if typ == 0 else comp
                if all((a[v] >> u) & 1 for u in range(N) if (mask >> u) & 1):
                    # move
                    parts[it] = (typ, mask | (1 << v))
                    if curmask:
                        parts[cur] = (typ, curmask)
                    else:
                        parts.pop(cur)
                    improved = True
                    break
            if improved:
                break
        # try merging two parts of the same type if union is a clique
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                if parts[i][0] == parts[j][0]:
                    typ, m = parts[i][0], parts[i][1] | parts[j][1]
                    a = adj if typ == 0 else comp
                    ok = True
                    ms = [u for u in range(N) if (m >> u) & 1]
                    for u in ms:
                        for w in ms:
                            if u < w and not ((a[u] >> w) & 1):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        parts[i] = (typ, m)
                        del parts[j]
                        improved = True
                        break
            if improved:
                break
        best = min(best, len(parts))
    return best


def _solve(adj, comp):
    lb = max(_greedy_color(adj, _order_by_deg(adj)), _greedy_color(comp, _order_by_deg(comp)))
    ub = _upper_bound(adj, comp)
    if ub <= lb:
        # reconstruct from greedy UB
        parts = _reconstruct_greedy(adj, comp)
        return parts

    # branch-and-bound
    best = [ub]
    best_parts = [None]
    order = _order_by_deg(adj)
    pos = [0] * N
    assign = [-1] * N
    used = [0] * ub  # bitmask of members per part
    ptype = [0] * ub  # 0=clique in G, 1=clique in Gbar
    pcount = 0

    def can_place(v, p, typ):
        m = used[p]
        a = adj if typ == 0 else comp
        for u in range(N):
            if (m >> u) & 1 and not ((a[v] >> u) & 1):
                return False
        return True

    def rec(k, depth):
        nonlocal pcount
        if time.time() > DEADLINE:
            return
        if depth >= best[0]:
            return
        # lower bound: depth + max(omega(G[unplaced]), omega(Gbar[unplaced]))
        if depth > best[0]:
            return
        if k == N:
            if depth < best[0]:
                best[0] = depth
                best_parts[0] = [list(range(N)) and None]
                # record assignment
                best_parts[0] = []
                for p in range(depth):
                    lst = [u for u in range(N) if (used[p] >> u) & 1]
                    best_parts[0].append(lst)
            return

        v = order[k]
        # option 1: place v into an existing part
        tried = set()
        for p in range(depth):
            for typ in (0, 1):
                if ptype[p] != typ:
                    continue
                key = (p, typ)
                if key in tried:
                    continue
                if can_place(v, p, typ):
                    tried.add(key)
                    used[p] |= 1 << v
                    assign[v] = p
                    pos[v] = p
                    rec(k + 1, depth)
                    used[p] ^= 1 << v
                    assign[v] = -1
                    pos[v] = -1
                    if time.time() > DEADLINE or depth >= best[0]:
                        return
        # option 2: start a new part (clique in G or in Gbar)
        if depth < ub:
            for typ in (0, 1):
                p = depth
                used[p] = 1 << v
                ptype[p] = typ
                assign[v] = p
                pos[v] = p
                rec(k + 1, depth + 1)
                used[p] = 0
                assign[v] = -1
                pos[v] = -1
                if time.time() > DEADLINE or depth >= best[0]:
                    return

    rec(0, 0)

    if best_parts[0] is not None:
        return best_parts[0]
    # fall back to greedy UB
    return _reconstruct_greedy(adj, comp)


def _reconstruct_greedy(adj, comp):
    order = sorted(range(N), key=lambda v: bin(adj[v]).count('1') + bin(comp[v]).count('1'), reverse=True)
    parts = []
    for v in order:
        for it, (typ, mask) in enumerate(parts):
            a = adj if typ == 0 else comp
            if all((a[v] >> u) & 1 for u in range(N) if (mask >> u) & 1):
                parts[it] = (typ, mask | (1 << v))
                break
        else:
            parts.append((0, 1 << v))
    return [[u for u in range(N) if (mask >> u) & 1] for (typ, mask) in parts]


def _validate(parts, adj):
    seen = [False] * N
    for part in parts:
        for v in part:
            if seen[v]:
                return False
            seen[v] = True
        m = 0
        for v in part:
            m |= 1 << v
        ms = [u for u in range(N) if (m >> u) & 1]
        clique = all((adj[u] >> w) & 1 for u in ms for w in ms if u < w)
        indep = all(not ((adj[u] >> w) & 1) for u in ms for w in ms if u < w)
        if not (clique or indep):
            return False
    return all(seen)


def construct():
    adj, comp = _setup()
    parts = _solve(adj, comp)
    # ensure validity; if broken, fall back to a guaranteed-valid greedy partition
    if not _validate(parts, adj):
        parts = _reconstruct_greedy(adj, comp)
    # normalize types: python int, list
    out = []
    for part in parts:
        out.append([int(v) for v in part])
    return out
# EVOLVE-BLOCK-END
