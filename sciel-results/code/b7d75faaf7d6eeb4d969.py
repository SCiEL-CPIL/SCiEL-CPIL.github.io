import time

N = 40
DEADLINE = 85  # safety margin inside the 90s evaluator window


def _edge_key(a, b):
    return (min(a, b), max(a, b))


def _build_edge_set(edges):
    s = set()
    for a, b in edges:
        s.add(_edge_key(a, b))
    return s


def _validate(edges, cycles):
    es = _build_edge_set(edges)
    if len(es) != len(edges):
        return False
    for cyc in cycles:
        L = len(cyc)
        if len(set(cyc)) != L:
            return False
        for i in range(L):
            if _edge_key(cyc[i], cyc[(i + 1) % L]) not in es:
                return False
    return True


def _greedy_remove(edges, cycles):
    """Repeatedly remove any edge not required by any cycle. Returns minimal list."""
    # Precompute required edges per cycle
    cyc_req = []
    for cyc in cycles:
        L = len(cyc)
        req = set()
        for i in range(L):
            req.add(_edge_key(cyc[i], cyc[(i + 1) % L]))
        cyc_req.append(req)

    changed = True
    while changed:
        changed = False
        new_edges = []
        for e in edges:
            key = _edge_key(*e)
            if any(key in r for r in cyc_req):
                new_edges.append(e)
            else:
                changed = True
        edges = new_edges
    return edges


def construct():
    t0 = time.time()

    # ---------- Build the two-layer (dual-fan) candidate graph ----------
    # Front backbone: 0-1-2-...-39  (39 edges)
    # Back  backbone: 0-4-8-12-16-20-24-28-32-36  (9 edges)
    # Spokes: (i,0) for i in {2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29,30,31,33,34,35,37,38,39}
    #   (i not in the back backbone and i != 1)
    # Total = 39 + 9 + 29 = 77 edges.

    back = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
    back_set = set(back)
    back_edges = [(back[i], back[i + 1]) for i in range(len(back) - 1)]
    front_edges = [(i, i + 1) for i in range(N - 1)]
    spoke_edges = [(i, 0) for i in range(2, N) if i not in back_set]

    edges = front_edges + back_edges + spoke_edges

    # De-duplicate
    seen = set()
    uniq = []
    for a, b in edges:
        if a == b:
            continue
        k = _edge_key(a, b)
        if k not in seen:
            seen.add(k)
            uniq.append((a, b))
    edges = uniq

    # ---------- Build cycles ----------
    # For each length k (3..40), a simple cycle using backbone + at most one spoke.
    # - If k-1 is in back: cycle = [0, b1, b2, ..., b_{k-1}, 0]  (all back edges, no spoke)
    # - Else: cycle = [0, 1, 2, ..., k-1, 0]  (front edges + spoke (k-1,0))
    cycles = []
    for k in range(3, N + 1):
        if (k - 1) in back_set:
            # find prefix of back ending at k-1
            idx = back.index(k - 1)
            cyc = back[:idx + 1]  # 0, b1, ..., b_{idx} = k-1
        else:
            cyc = list(range(k))  # 0,1,...,k-1
        cycles.append(cyc)

    # ---------- Validate ----------
    if not _validate(edges, cycles):
        # Fallback: pure fan (known valid)
        edges = [(i, i + 1) for i in range(N - 1)] + [(i, 0) for i in range(2, N)]
        cycles = [list(range(k)) for k in range(3, N + 1)]
        if not _validate(edges, cycles):
            # Should not happen; return as-is
            return {'edges': edges, 'cycles': cycles}

    # ---------- Greedy edge removal ----------
    edges = _greedy_remove(edges, cycles)

    # Final validation
    if not _validate(edges, cycles):
        # Rebuild from the validated pre-removal state (defensive)
        edges = front_edges + back_edges + spoke_edges
        seen = set()
        uniq = []
        for a, b in edges:
            k = _edge_key(a, b)
            if k not in seen:
                seen.add(k)
                uniq.append((a, b))
        edges = uniq
        edges = _greedy_remove(edges, cycles)

    return {'edges': edges, 'cycles': cycles}
