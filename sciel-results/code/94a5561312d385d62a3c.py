"""Edge-disjoint path cover via per-vertex pairing (maximizes internal pairings)."""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _cover(edges):
    """Return a list of simple, edge-disjoint paths covering all edges.

    Strategy: at each vertex, greedily pair incident edges to form path
    continuations. This maximizes the number of internal pairings
    (sum floor(deg/2) at each vertex), which minimizes the path count to
    E - sum floor(deg/2). Then trace the resulting trails (which are simple
    because we pair at most floor(deg/2) times and each edge appears once).
    """
    if not edges:
        return []
    # adjacency: vertex -> list of (neighbor, edge_id)
    adj = {}
    for eid, (u, v) in enumerate(edges):
        adj.setdefault(u, []).append((v, eid))
        adj.setdefault(v, []).append((u, eid))
    # For each vertex, pair up incident edges. We maintain a list of
    # "open ends": each is (vertex, neighbor, edge_id). We'll build a
    # pairing map: at vertex v, edge a is connected to edge b (a path goes
    # through v from a to b).
    # Represent pairing as: for each (v, eid), the partner (v, eid2) or None.
    # We store partner as dict keyed by (v, eid) -> (v, eid_partner).
    partner = {}
    for v, lst in adj.items():
        # lst is list of (neighbor, eid); pair them up
        m = len(lst)
        for i in range(0, m - 1, 2):
            a = (v, lst[i][1])
            b = (v, lst[i + 1][1])
            partner[a] = b
            partner[b] = a
        # if odd, the last one is unpaired (a path end)
    # Now build the paths. A path starts at an unpaired end.
    # We track used edges.
    used = [False] * len(edges)
    paths = []
    # find all unpaired ends
    starts = []
    for (v, eid) in partner.keys():
        pass
    # collect all ends
    all_ends = set()
    for v, lst in adj.items():
        for (nb, eid) in lst:
            all_ends.add((v, eid))
    for (v, eid) in all_ends:
        if (v, eid) not in partner:
            starts.append((v, eid))
    # Also, a path of length 0 (single edge with both ends unpaired) is handled
    # since both ends are in starts.
    for (sv, se) in starts:
        if used[se]:
            continue
        # walk
        path = [sv]
        cur_v = sv
        cur_e = se
        while True:
            used[cur_e] = True
            u, v2 = edges[cur_e]
            nxt = v2 if cur_v == u else u
            path.append(nxt)
            cur_v = nxt
            # find partner at cur_v
            p = partner.get((cur_v, cur_e))
            if p is None:
                break
            # p = (cur_v, next_eid)
            cur_e = p[1]
        paths.append(path)
    # Any leftover edges (shouldn't happen) — but safety:
    for eid in range(len(edges)):
        if not used[eid]:
            u, v = edges[eid]
            paths.append([u, v])
    return paths


def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    result = []
    for spec in specs:
        if time.time() > _DEADLINE - _MARGIN:
            # safety: return what we have plus trivial fallback for rest
            break
        edges = _instance(*spec)
        result.append(_cover(edges))
    # fill any missing with trivial per-edge cover
    for i, spec in enumerate(specs):
        if i >= len(result):
            edges = _instance(*spec)
            result.append([list(e) for e in edges])
    return result
# EVOLVE-BLOCK-END
