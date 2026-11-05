import time
from array import array

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    perm = array('I', range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [None] * n
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if seen[u] is not None:
            if v in seen[u]:
                return None
            seen[u].append(v)
        else:
            seen[u] = [v]
        adj[u].append(v)
        if seen[v] is not None:
            seen[v].append(u)
        else:
            seen[v] = [u]
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    a = _randrange(rng, m)
    b = _randrange(rng, m - 1)
    if b >= a:
        b += 1
    c = _randrange(rng, m - 2)
    if c >= a:
        c += 1
    if c >= b:
        c += 1
    d = _randrange(rng, m - 3)
    for x in (a, b, c):
        if d >= x:
            d += 1
    u1, v1 = stubs[a], stubs[b]
    u2, v2 = stubs[c], stubs[d]
    if u1 == v1 or u2 == v2:
        return False
    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
        return False
    adj[u1].remove(v1)
    adj[v1].remove(u1)
    adj[u2].remove(v2)
    adj[v2].remove(u2)
    if u1 == u2 or v1 == v2:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    if u2 in adj[u1] or v2 in adj[v1]:
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        return False
    adj[u1].append(u2)
    adj[u2].append(u1)
    adj[v1].append(v2)
    adj[v2].append(v1)
    return True

def _girth_fast(adj, g, n, deadline=None):
    """True iff no cycle shorter than g (exact, via non-backtracking walks)."""
    if n == 0:
        return True
    enc = n + 1
    for start in range(n):
        seen = [0] * (enc * enc)
        cur = array('I')
        for nb in adj[start]:
            s = start * enc + nb
            seen[s] = 1
            cur.append(s)
        depth = 1
        while depth <= g - 2:
            nxt = array('I')
            for s in cur:
                prev = s // enc
                curr = s - prev * enc
                for nx in adj[curr]:
                    if nx == prev:
                        continue
                    if nx == start and depth + 1 < g:
                        return False
                    ns = curr * enc + nx
                    if seen[ns] == 0:
                        seen[ns] = depth + 1
                        nxt.append(ns)
            cur = nxt
            if not cur:
                break
            depth += 1
            if deadline is not None and depth % 2 == 0 and time.time() > deadline:
                return False
    return True

def _short_cycle_pressure(adj, g, n, deadline=None):
    """Count of non-backtracking walks that close into a cycle of length < g."""
    if n == 0:
        return 0
    enc = n + 1
    total = 0
    for start in range(n):
        seen = [0] * (enc * enc)
        cur = array('I')
        for nb in adj[start]:
            s = start * enc + nb
            seen[s] = 1
            cur.append(s)
        depth = 1
        while depth <= g - 2:
            nxt = array('I')
            for s in cur:
                prev = s // enc
                curr = s - prev * enc
                for nx in adj[curr]:
                    if nx == prev:
                        continue
                    if nx == start:
                        if depth + 1 < g:
                            total += 1
                    else:
                        ns = curr * enc + nx
                        if seen[ns] == 0:
                            seen[ns] = depth + 1
                            nxt.append(ns)
            cur = nxt
            if not cur:
                break
            depth += 1
            if deadline is not None and time.time() > deadline:
                break
        if deadline is not None and time.time() > deadline:
            break
    return total

def construct(k, g):
    deadline = time.time() + 135.0
    rng = 0x9E3779B97F4A7C15

    def consider(n, adj):
        return _girth_fast(adj, g, n, deadline)

    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))

    n_start = max(2 * k, moore, 200)
    if n_start % 2 != 0:
        n_start += 1
    max_n = 520

    def emit(n, adj):
        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]

    n = n_start
    while n <= max_n and time.time() < deadline:
        if n % 2 != 0:
            n += 1
            continue
        adj = None
        attempts = 0
        while attempts < 8 and time.time() < deadline:
            cand = _pairing_regular(n, k, rng)
            if cand is not None:
                adj = cand
                break
            attempts += 1
        if adj is None:
            n += 2
            continue

        if consider(n, adj):
            return emit(n, adj)

        # --- 2-switch search biased toward removing short cycles ---
        p0 = _short_cycle_pressure(adj, g, n, deadline)
        if p0 == 0:
            return emit(n, adj)
        best_adj = None
        best_p = p0
        improved = False
        budget = deadline - 20.0
        # Phase 1: greedy accept any pressure reduction
        steps = 0
        while steps < 12000 and time.time() < budget:
            if _swap_regular(adj, n, k, rng):
                steps += 1
                p = _short_cycle_pressure(adj, g, n, deadline)
                if p < best_p:
                    best_p = p
                    best_adj = [lst[:] for lst in adj]
                    if p == 0:
                        improved = True
                        break
        if improved:
            return emit(n, best_adj)
        # Phase 2: annealed accept (smooth surrogate) to escape plateaus
        if best_p > 0 and time.time() < budget:
            adj = [lst[:] for lst in best_adj]
            cur_p = best_p
            temp = 3.0
            for it in range(12000):
                if time.time() > budget:
                    break
                if _swap_regular(adj, n, k, rng):
                    p = _short_cycle_pressure(adj, g, n, deadline)
                    if p <= cur_p:
                        cur_p = p
                        if p < best_p:
                            best_p = p
                            best_adj = [lst[:] for lst in adj]
                            if p == 0:
                                return emit(n, best_adj)
                    else:
                        # accept with probability 1/(1+exp((p-cur_p)/temp))
                        import math
                        if _randrange(rng, 1000000) < 1000000 // (1 + int(math.exp(min(20.0, (p - cur_p) / temp)))):
                            cur_p = p
            if best_p == 0:
                return emit(n, best_adj)
        n += 2

    # Fallback: LUW
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline - 5:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
