"""Rewritten: transitive base + careful local search to maximize distinct colors."""
import time
import random
import itertools
from collections import defaultdict

N = 16
TOTAL = N * (N - 1) // 2

VERTS_TO_EDGE = {}
CYCLES = []
EDGE_CYCLES = defaultdict(list)


def _build_structure():
    global CYCLES, VERTS_TO_EDGE, EDGE_CYCLES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            VERTS_TO_EDGE[(i, j)] = idx
            idx += 1
    for a, b, c, d in itertools.combinations(range(N), 4):
        CYCLES.append((VERTS_TO_EDGE[(a,b)], VERTS_TO_EDGE[(b,c)], VERTS_TO_EDGE[(c,d)], VERTS_TO_EDGE[(d,a)]))
        CYCLES.append((VERTS_TO_EDGE[(a,b)], VERTS_TO_EDGE[(b,d)], VERTS_TO_EDGE[(d,c)], VERTS_TO_EDGE[(c,a)]))
        CYCLES.append((VERTS_TO_EDGE[(a,c)], VERTS_TO_EDGE[(c,b)], VERTS_TO_EDGE[(b,d)], VERTS_TO_EDGE[(d,a)]))
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            EDGE_CYCLES[e].append(ci)

_build_structure()


def _count_distinct(colors):
    return len(set(colors))


def _is_valid(colors):
    for e1, e2, e3, e4 in CYCLES:
        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
            return False
    return True


def _build_base():
    colors = [0] * TOTAL
    for i in range(N):
        for j in range(i + 1, N):
            colors[VERTS_TO_EDGE[(i, j)]] = j - i - 1
    return colors


def construct():
    start_time = time.time()
    deadline = start_time + 50

    base = _build_base()
    colors = list(base)
    best = list(colors)
    best_distinct = _count_distinct(colors)

    rng = random.Random(42)
    max_color = max(colors)

    # Try to add new colors
    for step in range(200000):
        if time.time() > deadline:
            break

        e = rng.randint(0, TOTAL - 1)
        old_c = colors[e]
        new_c = max_color + 1

        # Check safety
        safe = True
        for ci in EDGE_CYCLES[e]:
            cyc = CYCLES[ci]
            others = {colors[ce] for ce in cyc if ce != e}
            if len(others) >= 3 and new_c not in others:
                safe = False
                break

        if safe:
            colors[e] = new_c
            max_color = new_c
            d = _count_distinct(colors)
            if d > best_distinct:
                best = list(colors)
                best_distinct = d
        else:
            # Try to fix by recoloring a conflicting edge
            colors[e] = new_c
            fixed = False
            for ci in EDGE_CYCLES[e]:
                cyc = CYCLES[ci]
                others = {colors[ce] for ce in cyc if ce != e}
                if len(others) >= 3 and new_c not in others:
                    # Pick one of the other edges to recolor
                    for target in [ce for ce in cyc if ce != e]:
                        old_tc = colors[target]
                        for match_target in [ce for ce in cyc if ce != target and ce != e]:
                            new_tc = colors[match_target]
                            if new_tc == old_tc:
                                continue
                            colors[target] = new_tc
                            ok = True
                            for ci2 in EDGE_CYCLES[target]:
                                if len({colors[ce] for ce in CYCLES[ci2]}) == 4:
                                    ok = False
                                    break
                            if ok:
                                fixed = True
                                break
                            else:
                                colors[target] = old_tc
                        if fixed:
                            break
                if fixed:
                    break

            if fixed and _is_valid(colors):
                max_color = new_c
                d = _count_distinct(colors)
                if d > best_distinct:
                    best = list(colors)
                    best_distinct = d
            else:
                colors[e] = old_c

    return best
