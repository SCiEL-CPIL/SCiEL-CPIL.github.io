"""Combined: fast bitmask K4/independent-6 detection + edge-flip local search."""
import time
import random

_DEADLINE = None

def _k4_found(adj, n):
    """Return a 4-tuple (a,b,c,d) forming a K4, or None. adj = list of int bitmasks."""
    for a in range(n):
        aa = adj[a]
        for b in range(a + 1, n):
            if not (aa >> b) & 1:
                continue
            ab = aa & adj[b]
            for c in range(b + 1, n):
                if not (ab >> c) & 1:
                    continue
                abc = ab & adj[c]
                for d in range(c + 1, n):
                    if (abc >> d) & 1:
                        return (a, b, c, d)
    return None

def _ind6_found(adj, n):
    """Return a 6-tuple forming an independent set, or None. adj = list of int bitmasks."""
    allmask = (1 << n) - 1
    for a in range(n):
        na = allmask ^ (1 << a) ^ adj[a]
        for b in range(a + 1, n):
            if not (na >> b) & 1:
                continue
            nab = na & (allmask ^ (1 << b) ^ adj[b])
            for c in range(b + 1, n):
                if not (nab >> c) & 1:
                    continue
                nabc = nab & (allmask ^ (1 << c) ^ adj[c])
                for d in range(c + 1, n):
                    if not (nabc >> d) & 1:
                        continue
                    nabcd = nabc & (allmask ^ (1 << d) ^ adj[d])
                    for e in range(d + 1, n):
                        if not (nabcd >> e) & 1:
                            continue
                        if nabcd & (allmask ^ (1 << e) ^ adj[e]) & ((1 << n) - 1):
                            # need one more vertex after e
                            rest = nabcd & (allmask ^ (1 << e) ^ adj[e])
                            # pick smallest bit greater than e
                            higher = rest & (~((1 << (e + 1)) - 1))
                            if higher:
                                f = (higher & -higher).bit_length() - 1
                                return (a, b, c, d, e, f)
    return None

def _is_valid(adj, n):
    return _k4_found(adj, n) is None and _ind6_found(adj, n) is None

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        m = adj[i]
        j = i + 1
        while m:
            if j >= n:
                break
            if (m >> j) & 1:
                edges.append([i, j])
            m &= m - 1
            j = (m & -m).bit_length() - 1 if m else n
    # simpler robust build
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def _local_search(n, p, deadline):
    """Try to find a valid graph on n vertices via edge flips."""
    allmask = (1 << n) - 1
    for attempt in range(60):
        if time.time() > deadline:
            return None
        random.seed(1000 + attempt)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        for step in range(40000):
            if time.time() > deadline:
                return None
            k4 = _k4_found(adj, n)
            if k4 is None:
                ind = _ind6_found(adj, n)
                if ind is None:
                    return adj
                # break independent set by adding an edge inside it
                a, b, c, d, e, f = ind
                pairs = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
                u, v = random.choice(pairs)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                a, b, c, d = k4
                pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
                u, v = random.choice(pairs)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    return None

def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 112  # safety margin before 150s kill
    random.seed(42)

    for n in range(36, 16, -1):
        if time.time() > _DEADLINE:
            break
        # edge probability near 0.5
        adj = _local_search(n, 0.5, _DEADLINE)
        if adj is not None:
            return {'n': n, 'edges': _edges_from_adj(adj, n)}

    # Fallback: Paley graph on 17 vertices (known K4-free, alpha=5)
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
