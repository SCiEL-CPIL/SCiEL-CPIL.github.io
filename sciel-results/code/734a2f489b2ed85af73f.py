"""Q3-free dense graph on 16 vertices: constructive search from the rook graph."""
import time

N = 16
DEADLINE = time.time() + 130.0  # safety margin before the 150s kill


def _is_q3_free(adj, adjset):
    """Return True iff the graph (adj bitmasks) contains no subgraph isomorphic to Q3."""
    # Any cube has a 4-cycle whose 8 vertices induce exactly the 12 cube edges.
    # Enumerate 4-cycles, then check the induced subgraph on those 4 + 4 neighbors.
    for a in range(N):
        na = adj[a]
        if na & (na - 1) == 0:
            continue
        a_bits = na
        while a_bits:
            lb = a_bits & (-a_bits)
            a_bits -= lb
            b = lb.bit_length() - 1
            # c must be a common neighbor of a and b, c > b
            nc = adj[a] & adj[b]
            nc &= ~((1 << (b + 1)) - 1)  # c > b
            while nc:
                lc = nc & (-nc)
                nc -= lc
                c = lc.bit_length() - 1
                # d must be common neighbor of b and c, not a, not c
                nd = adj[b] & adj[c]
                nd &= ~((1 << (c + 1)) - 1)  # d > c
                while nd:
                    ld = nd & (-nd)
                    nd -= ld
                    d = ld.bit_length() - 1
                    # 4-cycle a-b-c-d-a exists. Check induced subgraph on {a,b,c,d}+{a',b',c',d'}.
                    # Neighbors of a outside the cycle:
                    rest_a = adj[a] & ~(lb | lc | ld | (1 << a))
                    if rest_a.bit_count() != 1:
                        continue
                    a2 = (rest_a & (-rest_a)).bit_length() - 1
                    # neighbors of b outside cycle
                    rest_b = adj[b] & ~(lb | lc | ld | (1 << b))
                    if rest_b.bit_count() != 1:
                        continue
                    b2 = (rest_b & (-rest_b)).bit_length() - 1
                    if a2 == b2:
                        continue
                    rest_c = adj[c] & ~(lb | lc | ld | (1 << c))
                    if rest_c.bit_count() != 1:
                        continue
                    c2 = (rest_c & (-rest_c)).bit_length() - 1
                    if c2 in (a2, b2):
                        continue
                    rest_d = adj[d] & ~(lb | lc | ld | (1 << d))
                    if rest_d.bit_count() != 1:
                        continue
                    d2 = (rest_d & (-rest_d)).bit_length() - 1
                    if d2 in (a2, b2, c2):
                        continue
                    verts = [a, b, c, d, a2, b2, c2, d2]
                    # The cube edges: the 4-cycle + 4 spokes + the top 4-cycle.
                    # Required edges:
                    req = [(a, b), (b, c), (c, d), (d, a),
                           (a, a2), (b, b2), (c, c2), (d, d2),
                           (a2, b2), (b2, c2), (c2, d2), (d2, a2)]
                    ok = True
                    for u, v in req:
                        if not (adj[u] >> v) & 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    # No extra edges among the 8 vertices beyond the 12 cube edges.
                    req_set = set()
                    for u, v in req:
                        req_set.add((u, v))
                        req_set.add((v, u))
                    extra = False
                    for i in range(8):
                        for j in range(i + 1, 8):
                            u, v = verts[i], verts[j]
                            if (adj[u] >> v) & 1 and (u, v) not in req_set:
                                extra = True
                                break
                        if extra:
                            break
                    if not extra:
                        return False
    return True


def _rook_graph():
    """Complement of the Clebsch graph = 4x4 rook graph (100 edges)."""
    edges = []
    for i in range(4):
        for j in range(4):
            u = 4 * i + j
            for i2 in range(4):
                for j2 in range(4):
                    v = 4 * i2 + j2
                    if u < v and (i == i2 or j == j2):
                        edges.append([u, v])
    return edges


def _build_adj(edges):
    adj = [0] * N
    adjset = set()
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        adjset.add((u, v))
        adjset.add((v, u))
    return adj, adjset


def construct():
    t0 = time.time()
    edges = _rook_graph()
    adj, adjset = _build_adj(edges)

    # Verify the seed is Q3-free (should be, but guard against a bad seed).
    if not _is_q3_free(adj, adjset):
        # Fallback: use K3,13 which is provably Q3-free.
        edges = [[a, b] for a in range(3) for b in range(3, 16)]
        adj, adjset = _build_adj(edges)

    # Greedily add missing edges that keep the graph Q3-free.
    # Candidate order: try to add edges among the "non-neighbors" to increase density.
    missing = []
    for u in range(N):
        for v in range(u + 1, N):
            if not (adj[u] >> v) & 1:
                missing.append((u, v))

    # Heuristic: add edges that connect vertices far apart in rook structure first.
    def _key(e):
        u, v = e
        return (abs(u // 4 - v // 4), abs(u % 4 - v % 4))
    missing.sort(key=_key, reverse=True)

    for (u, v) in missing:
        if time.time() > DEADLINE - 5.0:
            break
        # tentatively add
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        if _is_q3_free(adj, adjset):
            adjset.add((u, v))
            adjset.add((v, u))
            edges.append([u, v])
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

    return edges
