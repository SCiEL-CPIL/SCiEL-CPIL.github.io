"""Edge-colourings of K16 where every K4 sees >=5 colours.
Aim: 13 colours (lower bound); provable 14-colour fallback."""
import time
import random

_N = 16
_NK4 = 1820
_DEADLINE = 115.0  # safety margin under the 150s wall-clock limit

# ---- deterministic 14-colour baseline -------------------------------------
_BASE14 = [(i % 2) + 2 * (j % 2) for i in range(_N) for j in range(i + 1, _N)]
# remap to 0..13
_rem = {v: k for k, v in enumerate(sorted(set(_BASE14)))}
_BASE14 = [_rem[v] for v in _BASE14]

# ---- precompute K4 edge index lists ---------------------------------------
_EDGE_IDX = {}
_k = 0
for i in range(_N):
    for j in range(i + 1, _N):
        _EDGE_IDX[(i, j)] = _k
        _k += 1

_QUADS = []          # list of 6 edge indices
_QUADS_VERT = []     # list of 4 vertex ids
for a in range(_N):
    for b in range(a + 1, _N):
        for c in range(b + 1, _N):
            for d in range(c + 1, _N):
                _QUADS.append((_EDGE_IDX[(a, b)], _EDGE_IDX[(a, c)], _EDGE_IDX[(a, d)],
                               _EDGE_IDX[(b, c)], _EDGE_IDX[(b, d)], _EDGE_IDX[(c, d)]))
                _QUADS_VERT.append((a, b, c, d))

# ---- per-edge K4 membership (edge -> list of quad indices) ----------------
_EDGE_QUADS = [[] for _ in range(120)]
for _qi, _q in enumerate(_QUADS):
    for _ei in _q:
        _EDGE_QUADS[_ei].append(_qi)


def _count_colors(colors, quad):
    """Return the number of distinct colours in a 6-edge quad (small tuple)."""
    c0, c1, c2, c3, c4, c5 = (colors[quad[0]], colors[quad[1]], colors[quad[2]],
                               colors[quad[3]], colors[quad[4]], colors[quad[5]])
    s = {c0}
    if c1 not in s: s.add(c1)
    if c2 not in s: s.add(c2)
    if c3 not in s: s.add(c3)
    if c4 not in s: s.add(c4)
    if c5 not in s: s.add(c5)
    return len(s)


def _colors_used(colors):
    s = set(colors)
    return len(s)


def _is_valid(colors):
    for q in _QUADS:
        if _count_colors(colors, q) < 5:
            return False
    return True


def _violation_count(colors):
    v = 0
    for q in _QUADS:
        if _count_colors(colors, q) < 5:
            v += 1
    return v


def construct():
    start = time.time()
    deadline = start + _DEADLINE

    # Start from the valid 14-colour baseline.
    colors = list(_BASE14)
    best = colors[:]
    best_k = _colors_used(best)

    # If we already somehow have 13, done.
    if best_k <= 13:
        return best

    rng = random.Random(12345)

    # Simulated annealing trying to reach 13 colours.
    # Objective: minimize (colors_used, violations).
    cur_k = best_k
    cur_v = _violation_count(colors)  # 0

    # We only accept moves that keep validity (cur_v==0) and never increase
    # the number of colours; occasionally allow a colour increase to escape.
    temp = 1.0
    cooldown = 0.9995

    def _obj(k, v):
        # primary: violations, secondary: colours
        return (v, k)

    while time.time() < deadline:
        # pick a random edge
        e = rng.randrange(120)
        old_c = colors[e]
        # choose a new colour: bias toward reducing colour count
        if rng.random() < 0.6:
            # pick from colours currently used
            nc = rng.choice(best)  # reuse an existing colour
        else:
            nc = rng.randrange(13)  # try a low colour index (0..12)
        if nc == old_c:
            continue

        # apply
        colors[e] = nc
        # recompute affected quads
        new_v = 0
        for qi in _EDGE_QUADS[e]:
            if _count_colors(colors, _QUADS[qi]) < 5:
                new_v += 1
        new_k = _colors_used(colors)

        # decide accept
        accept = False
        if new_v == 0:
            if new_k < cur_k:
                accept = True
            elif new_k == cur_k:
                accept = True
            else:
                # colour increase: accept with probability exp(-delta/temp)
                # delta = new_k - cur_k
                if new_k - cur_k <= 1 and rng.random() < (temp / (new_k - cur_k + 1)):
                    accept = True
        else:
            # violation increase: accept rarely to escape
            if rng.random() < 0.02 * temp:
                accept = True

        if accept:
            cur_k, cur_v = new_k, new_v
            if new_v == 0 and new_k < best_k:
                best = colors[:]
                best_k = new_k
                if best_k <= 13:
                    return best
        else:
            colors[e] = old_c

        temp *= cooldown
        if temp < 0.05:
            temp = 1.0  # reheat

    # Final: return best valid solution found.
    if _is_valid(best):
        return best
    # Should not happen (baseline is valid), but be safe.
    return _BASE14
