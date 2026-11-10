# EVOLVE-BLOCK-START
import time
import random

_N = 16
_EDGES = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_EDGE_INDEX = {(i, j): k for k, (i, j) in enumerate(_EDGES)}
_QUADS = []
for a in range(_N):
    for b in range(a + 1, _N):
        for c in range(b + 1, _N):
            for d in range(c + 1, _N):
                _QUADS.append((a, b, c, d))

# Precompute, for each edge, the list of quads containing it.
_EDGE_QUADS = [[] for _ in range(len(_EDGES))]
for q in _QUADS:
    a, b, c, d = q
    for (u, v) in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
        _EDGE_QUADS[_EDGE_INDEX[(u, v)]].append(q)


def _quad_colours(colours, q):
    a, b, c, d = q
    return (
        colours[_EDGE_INDEX[(a, b)]],
        colours[_EDGE_INDEX[(a, c)]],
        colours[_EDGE_INDEX[(a, d)]],
        colours[_EDGE_INDEX[(b, c)]],
        colours[_EDGE_INDEX[(b, d)]],
        colours[_EDGE_INDEX[(c, d)]],
    )


def _count_violations(colours):
    bad = 0
    for q in _QUADS:
        cs = _quad_colours(colours, q)
        # need at least 5 distinct
        # quick check: number of distinct < 5
        if len(set(cs)) < 5:
            bad += 1
    return bad


def _palette_size(colours):
    return len(set(colours))


def _base_colours():
    return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]


def construct():
    deadline = time.time() + 130.0  # safety margin under 150s
    rng = random.Random(12345)

    colours = _base_colours()
    best = list(colours)
    best_viol = _count_violations(colours)
    best_pal = _palette_size(colours)
    # objective: (violations, palette)
    best_key = (best_viol, best_pal)

    def key_of(c):
        v = _count_violations(c)
        if v != 0:
            return (v, 0)
        return (0, _palette_size(c))

    # We want violations==0 then minimise palette.
    # Local search: pick an edge, try a new colour in a bounded palette.
    # Palette bound: we try to push palette down. Start with 17, try to shrink.
    # Approach: iterate target palette T from 15 downward; but simpler:
    # do a search that tries to reduce palette while keeping 0 violations.

    n_edges = len(colours)

    def try_step(c, edge_idx, new_col, T):
        # only allow colours in [0, T)
        old = c[edge_idx]
        if old == new_col:
            return False
        c[edge_idx] = new_col
        # check affected quads
        ok = True
        for q in _EDGE_QUADS[edge_idx]:
            cs = _quad_colours(c, q)
            if len(set(cs)) < 5:
                ok = False
                break
        if ok:
            return True
        c[edge_idx] = old
        return False

    # First, ensure we have a valid 17-colour solution (we do).
    # Now try to reduce palette: for T from 15 down to 13, try to map into [0,T).
    # Strategy: repeatedly, for each edge, if its colour >= T, change to a colour
    # in [0,T) that keeps validity (prefer one that keeps palette small).
    # Also allow "swap" moves to free up palette.

    # We'll run a multi-target search. For each target T, attempt to reach
    # a valid colouring using only colours in [0,T).
    for T in (15, 14, 13):
        if time.time() > deadline - 5.0:
            break
        c = list(colours)
        # initial projection: map colour x -> x % T (may create violations)
        for e in range(n_edges):
            c[e] = c[e] % T
        # Now hill-climb to remove violations, keeping colours in [0,T).
        # Track violations count.
        viol = _count_violations(c)
        # We need viol == 0. Use local search.
        # Represent violation count; do random edge-colour changes that reduce viol.
        # Precompute per-quad distinct count quickly via recomputation on affected quads.
        # We'll keep an array of quad distinct counts.
        quad_dist = [len(set(_quad_colours(c, q))) for q in _QUADS]
        viol = sum(1 for x in quad_dist if x < 5)

        def recompute_edge(e):
            # returns delta violation
            nonlocal viol
            # remove old contribution
            for q in _EDGE_QUADS[e]:
                qi = _QUAD_INDEX[q]
                quad_dist[qi] = 0  # mark dirty
            # recompute
            for q in _EDGE_QUADS[e]:
                qi = _QUAD_INDEX[q]
                nd = len(set(_quad_colours(c, q)))
                quad_dist[qi] = nd
                if nd < 5:
                    viol += 1
            # NOTE: the above double counts; do it properly below.
        # Simpler: recompute affected quads properly.
        def set_edge(e, newc):
            nonlocal viol
            old = c[e]
            if old == newc:
                return
            # subtract old
            for q in _EDGE_QUADS[e]:
                qi = _QUAD_INDEX[q]
                if quad_dist[qi] < 5:
                    viol -= 1
                quad_dist[qi] = -1
            c[e] = newc
            for q in _EDGE_QUADS[e]:
                qi = _QUAD_INDEX[q]
                nd = len(set(_quad_colours(c, q)))
                quad_dist[qi] = nd
                if nd < 5:
                    viol += 1

        # Build quad index map
        _QUAD_INDEX = {q: i for i, q in enumerate(_QUADS)}

        improved = True
        steps = 0
        while improved and viol > 0 and time.time() < deadline - 5.0:
            improved = False
            steps += 1
            # try to fix: pick a violating quad, change one of its edges to a
            # colour in [0,T) that increases its distinct count without hurting others.
            # We'll do randomised: pick random edge, random colour in [0,T),
            # accept if viol does not increase (and prefer decrease).
            e = rng.randrange(n_edges)
            newc = rng.randrange(T)
            old_viol = viol
            set_edge(e, newc)
            if viol < old_viol:
                improved = True
            elif viol == old_viol and rng.random() < 0.5:
                improved = True
            else:
                # revert
                set_edge(e, colours_e_old) if False else None
                # need old value; redo: store before
                pass
        # The revert logic above is buggy; do a cleaner loop.
        # Restart with cleaner implementation.
        break

    # Fallback: return best found (the valid 17-colour base).
    # But we can do better with a clean local search below.
    # ---- Clean local search to minimise palette with 0 violations ----
    c = list(colours)
    quad_dist = [len(set(_quad_colours(c, q))) for q in _QUADS]
    viol = sum(1 for x in quad_dist if x < 5)
    quad_index = {q: i for i, q in enumerate(_QUADS)}

    def do_set(e, newc):
        nonlocal viol
        old = c[e]
        if old == newc:
            return
        for q in _EDGE_QUADS[e]:
            qi = quad_index[q]
            if quad_dist[qi] < 5:
                viol -= 1
            quad_dist[qi] = -1
        c[e] = newc
        for q in _EDGE_QUADS[e]:
            qi = quad_index[q]
            nd = len(set(_quad_colours(c, q)))
            quad_dist[qi] = nd
            if nd < 5:
                viol += 1

    # Try to reduce palette: for T in 15..13, project then fix.
    for T in (15, 14, 13):
        if time.time() > deadline - 8.0:
            break
        c2 = list(c)
        for e in range(n_edges):
            c2[e] = c2[e] % T
        qd = [len(set(_quad_colours(c2, q))) for q in _QUADS]
        v = sum(1 for x in qd if x < 5)
        # local search to clear violations
        it = 0
        stuck = 0
        while v > 0 and it < 200000 and time.time() < deadline - 8.0:
            it += 1
            e = rng.randrange(n_edges)
            newc = rng.randrange(T)
            oldv = v
            # apply
            old = c2[e]
            for q in _EDGE_QUADS[e]:
                qi = quad_index[q]
                if qd[qi] < 5:
                    v -= 1
                qd[qi] = -1
            c2[e] = newc
            for q in _EDGE_QUADS[e]:
                qi = quad_index[q]
                nd = len(set(_quad_colours(c2, q)))
                qd[qi] = nd
                if nd < 5:
                    v += 1
            if v < oldv:
                stuck = 0
            elif v == oldv:
                stuck += 1
            else:
                # revert
                for q in _EDGE_QUADS[e]:
                    qi = quad_index[q]
                    if qd[qi] < 5:
                        v -= 1
                    qd[qi] = -1
                c2[e] = old
                for q in _EDGE_QUADS[e]:
                    qi = quad_index[q]
                    nd = len(set(_quad_colours(c2, q)))
                    qd[qi] = nd
                    if nd < 5:
                        v += 1
                stuck += 1
            if stuck > 2000:
                # restart projection
                for e2 in range(n_edges):
                    c2[e2] = _base_colours()[e2] % T
                qd = [len(set(_quad_colours(c2, q))) for q in _QUADS]
                v = sum(1 for x in qd if x < 5)
                stuck = 0
        if v == 0:
            c = list(c2)
            pal = _palette_size(c)
            if pal < best_pal:
                best = list(c)
                best_pal = pal
            # try smaller T next
            continue
        else:
            # could not fix for this T; try next (smaller) T may be harder; break
            break

    # Final: if best is valid, return it; else return base.
    if _count_violations(best) == 0:
        return best
    return _base_colours()
# EVOLVE-BLOCK-END
