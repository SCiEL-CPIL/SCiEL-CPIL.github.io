import random
import time

def _has_c6(u, v, adj_bits, n):
    """
    Check if adding edge (u, v) creates a C6, i.e. is there a simple path
    of length 5 from u to v: u -> a -> b -> c -> d -> v, all distinct.
    Fast bitmask-based check.
    """
    # d in N(v), d != u, d != v
    for d in range(n):
        if d == u or d == v:
            continue
        if not ((adj_bits[v] >> d) & 1):
            continue
        # need simple path u -> a -> b -> c -> d avoiding v
        # a in N(u), a not in {u, d, v}
        for a in range(n):
            if a == u or a == d or a == v:
                continue
            if not ((adj_bits[u] >> a) & 1):
                continue
            # b in N(a), b not in {u, a, d, v}
            for b in range(n):
                if b == u or b == a or b == d or b == v:
                    continue
                if not ((adj_bits[a] >> b) & 1):
                    continue
                # c in N(b), c not in {u, a, b, d, v}; and c adjacent to d
                # c must be in N(b) and N(d)
                # iterate over N(b)
                cb = adj_bits[b]
                cd = adj_bits[d]
                common = cb & cd
                # c must be in common, c != u,a,b,d,v
                # test each candidate c in common
                m = common
                while m:
                    c = (m & -m).bit_length() - 1
                    m &= m - 1
                    if c == u or c == a or c == b or c == d or c == v:
                        continue
                    # c in N(b) (guaranteed by common) and c in N(d) (guaranteed)
                    return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n = 40
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # ---- Build initial graph: 8 disjoint K5s ----
    adj_bits = [0] * n
    edge_set = set()
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edge_set.add((u, v))
                edges.append((u, v))

    def snapshot():
        return (len(edges), [b for b in adj_bits])

    best_count, best_bits = snapshot()

    def refill():
        """Add any edge that doesn't create a C6, repeatedly, until none remain."""
        nonlocal edges, adj_bits, edge_set
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u, v in all_pairs:
                if (u, v) in edge_set:
                    continue
                if not _has_c6(u, v, adj_bits, n):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edge_set.add((u, v))
                    edges.append((u, v))
                    changed = True
                    if len(edges) > best_count:
                        best_count, best_bits = snapshot()
                    if time.time() > deadline:
                        return

    # ---- Local search with kicks (aggressive perturbation when stuck) ----
    while time.time() < deadline - 5.0:
        refill()

        # If we can't add anything, kick: remove a few random edges
        # Aggressive multi-edge perturbation (1..4) to escape local optima.
        kick_size = random.randint(1, 4)
        for _ in range(kick_size):
            if not edges:
                break
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            if (u, v) not in edge_set:
                continue
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            edge_set.discard((u, v))
            edges.pop(idx)
        if time.time() > deadline - 5.0:
            break

    # ---- Final polish: one more refill on the current best structure ----
    if time.time() < deadline - 2.0:
        # Restore the best snapshot we found, then do a final greedy pass
        adj_bits = list(best_bits)
        # rebuild edge_set and edges from best_bits
        edge_set = set()
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    edge_set.add((u, v))
                    edges.append((u, v))
        best_count = len(edges)
        refill()

    # Rebuild final edge list from best snapshot
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_bits[u] >> v) & 1:
                final_edges.append([u, v])

    return final_edges
