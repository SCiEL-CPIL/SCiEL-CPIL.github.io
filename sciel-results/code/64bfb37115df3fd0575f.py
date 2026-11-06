"""Dense rainbow-4-cycle edge-coloured graph on 20 vertices, 20 colours.
Strategy: randomized local search.
  - Base: cycle C20 with a known-valid rainbow-4-cycle colouring (c(u,v)=(u+v)%20).
  - Repair: when a 4-cycle has a colour collision, recolor one of its edges (cheap, O(n^2) check).
  - Grow: repeatedly try to add a missing edge with a colour that keeps every 4-cycle rainbow.
  - Deadline guard returns the best valid result found.
"""
import time
import random

N = 20
C = 20
DEADLINE = 115.0  # safety margin under 120s

# adjacency: adj[u][v] = colour or -1 if absent
# We keep a dict keyed by (min(u,v), max(u,v)) -> colour for fast lookup.


def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)


def _build_base_cycle():
    """Cycle 0-1-2-...-19-0 with colour (u+v)%20. Every 4-cycle is rainbow."""
    edges = {}
    for i in range(N):
        j = (i + 1) % N
        u, v = min(i, j), max(i, j)
        edges[(u, v)] = (i + j) % C
    return edges


def _all_4cycles():
    """Yield each 4-cycle once as a frozenset of 4 edge keys.
    A 4-cycle is determined by 4 vertices a<b<c<d with edges forming a 4-cycle.
    We enumerate by choosing 4 vertices and the 3 possible 4-cycle structures
    (a-b-c-d-a, a-b-d-c-a, a-c-b-d-a).
    """
    from itertools import combinations
    verts = list(range(N))
    cycles = []
    for a, b, c, d in combinations(verts, 4):
        # three 4-cycles on 4 vertices
        for (e1, e2, e3, e4) in [
            ((a, b), (b, c), (c, d), (d, a)),
            ((a, b), (b, d), (d, c), (c, a)),
            ((a, c), (c, b), (b, d), (d, a)),
        ]:
            keys = frozenset(_edge_key(*e) for e in (e1, e2, e3, e4))
            if len(keys) == 4:  # all four edges distinct
                cycles.append(keys)
    return cycles


def _colors_of_cycle(edges, cycle):
    """Return list of colours of the 4 edges in the cycle, or None if any edge missing."""
    cols = []
    for k in cycle:
        c = edges.get(k, -1)
        if c < 0:
            return None
        cols.append(c)
    return cols


def _is_valid(edges, all_cycles):
    for cyc in all_cycles:
        cols = _colors_of_cycle(edges, cyc)
        if cols is None:
            continue  # not a subgraph
        if len(set(cols)) != 4:
            return False
    return True


def _violations_for_edge(edges, all_cycles, k):
    """Return list of cycles (as frozensets) that contain edge k and are violated."""
    out = []
    for cyc in all_cycles:
        if k in cyc:
            cols = _colors_of_cycle(edges, cyc)
            if cols is None:
                continue
            if len(set(cols)) != 4:
                out.append(cyc)
    return out


def _try_add_edge(edges, all_cycles, u, v, rng):
    """Try to add edge (u,v) with some colour that keeps validity. Returns True if added."""
    k = _edge_key(u, v)
    if k in edges:
        return False
    # candidate colours: prefer ones not already used by neighbours' edges to reduce collisions
    used = set()
    for (a, b), c in edges.items():
        if a in (u, v) or b in (u, v):
            used.add(c)
    cand = [c for c in range(C) if c not in used]
    if not cand:
        cand = list(range(C))
    rng.shuffle(cand)
    for c in cand:
        edges[k] = c
        if _is_valid(edges, all_cycles):
            return True
        else:
            del edges[k]
    return False


def _try_recolor(edges, all_cycles, k, rng):
    """Try to recolor edge k to fix its violations. Returns True if improved."""
    if k not in edges:
        return False
    old = edges[k]
    # collect colours that cause violations
    bad = set()
    for cyc in all_cycles:
        if k in cyc:
            cols = _colors_of_cycle(edges, cyc)
            if cols is None:
                continue
            if len(set(cols)) != 4:
                # the offending colours are those that repeat in this cycle
                seen = set()
                for cc in cols:
                    if cc in seen:
                        bad.add(cc)
                    seen.add(cc)
    cand = [c for c in range(C) if c not in bad and c != old]
    if not cand:
        cand = [c for c in range(C) if c != old]
    rng.shuffle(cand)
    for c in cand:
        edges[k] = c
        if _is_valid(edges, all_cycles):
            return True
    edges[k] = old
    return False


def _try_remove_edge(edges, all_cycles, k):
    if k not in edges:
        return False
    del edges[k]
    return True


def construct():
    t0 = time.time()
    rng = random.Random(20240517)

    all_cycles = _all_4cycles()

    # Start from the valid cycle base.
    edges = _build_base_cycle()
    best = dict(edges)
    best_count = len(edges)

    def remaining():
        return DEADLINE - (time.time() - t0)

    # Phase 1: greedy add edges (random order), keep if valid.
    missing = [(_edge_key(u, v)) for u in range(N) for v in range(u + 1, N) if (u, v) not in edges]
    rng.shuffle(missing)
    for k in missing:
        if remaining() < 5.0:
            break
        if _try_add_edge(edges, all_cycles, k[0], k[1], rng):
            if len(edges) > best_count:
                best = dict(edges)
                best_count = len(edges)

    # Phase 2: local search — recolor to fix, then add more edges.
    # We iterate: find a violated cycle, recolor one of its edges; if graph becomes
    # valid with more edges than before, keep. Otherwise try adding a new edge.
    # Since base is valid and we only add validly, edges should stay valid.
    # To grow further, we do random restarts of the add-phase with different orderings.
    for restart in range(200):
        if remaining() < 8.0:
            break
        # random order of all missing edges
        missing = [(_edge_key(u, v)) for u in range(N) for v in range(u + 1, N) if (u, v) not in edges]
        rng.shuffle(missing)
        added_this = 0
        for k in missing:
            if remaining() < 8.0:
                break
            if _try_add_edge(edges, all_cycles, k[0], k[1], rng):
                added_this += 1
                if len(edges) > best_count:
                    best = dict(edges)
                    best_count = len(edges)
        # Occasionally try to recolor a random edge to free up a colour, then re-add.
        if remaining() < 8.0:
            break
        if edges:
            ks = list(edges.keys())
            rng.shuffle(ks)
            for k in ks[:10]:
                if remaining() < 8.0:
                    break
                old = edges[k]
                # temporarily remove
                del edges[k]
                # try to recolor-and-keep
                if _try_recolor(edges, all_cycles, k, rng):
                    pass  # recolored in place
                else:
                    edges[k] = old
                # now try to add a new edge
                missing2 = [(_edge_key(u, v)) for u in range(N) for v in range(u + 1, N) if (u, v) not in edges]
                if missing2:
                    rng.shuffle(missing2)
                    for k2 in missing2[:50]:
                        if remaining() < 8.0:
                            break
                        if _try_add_edge(edges, all_cycles, k2[0], k2[1], rng):
                            if len(edges) > best_count:
                                best = dict(edges)
                                best_count = len(edges)
                            break

    if remaining() < 1.0:
        return [[u, v, best[(u, v)]] for (u, v) in best]

    # Final: ensure validity of best (it was only kept when valid).
    assert _is_valid(best, all_cycles)
    return [[u, v, best[(u, v)]] for (u, v) in best]


if __name__ == "__main__":
    res = construct()
    print(len(res))
