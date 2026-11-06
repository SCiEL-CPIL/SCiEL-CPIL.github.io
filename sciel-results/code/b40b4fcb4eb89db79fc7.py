import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 112.0          # hard safety margin (evaluator kills at 150s)
    n = 16
    num_edges = 120

    # edge index map
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            idx += 1

    # all K4s and their 6 edge indices
    k4s = []
    for a, b, c, d in combinations(range(n), 4):
        k4s.append([edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
                    edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]])

    # per-edge list of K4 indices
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, eidx in enumerate(k4s):
        for ei in eidx:
            edge_to_k4s[ei].append(k4_idx)

    def distinct_count(eidx, colors):
        c0 = colors[eidx[0]]; c1 = colors[eidx[1]]; c2 = colors[eidx[2]]
        c3 = colors[eidx[3]]; c4 = colors[eidx[4]]; c5 = colors[eidx[5]]
        d = 1
        if c1 != c0: d += 1
        if c2 != c0 and c2 != c1: d += 1
        if c3 != c0 and c3 != c1 and c3 != c2: d += 1
        if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: d += 1
        if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: d += 1
        return d

    def count_violations(colors):
        return sum(1 for eidx in k4s if distinct_count(eidx, colors) < 5)

    def count_bad4(colors):
        return sum(1 for eidx in k4s if distinct_count(eidx, colors) == 4)

    def valid_move(ei, new_c, colors, skip_k4=-1):
        for k in edge_to_k4s[ei]:
            if k == skip_k4:
                continue
            if distinct_count(k4s[k], colors) < 5:
                return False
        return True

    # ---- Phase 1: repair a random 4-colouring to validity ----
    best = None
    rng = random.Random(12345)
    for trial in range(6):
        if time.time() > deadline - 40:
            break
        colors = [rng.randint(0, 3) for _ in range(num_edges)]
        viol = count_violations(colors)
        stuck = 0
        while viol > 0 and time.time() < deadline - 40:
            # find a violating K4
            vk = -1
            for k in range(len(k4s)):
                if distinct_count(k4s[k], colors) < 5:
                    vk = k
                    break
            if vk < 0:
                viol = 0
                break
            eidx = k4s[vk]
            fixed = False
            for ei in eidx:
                old = colors[ei]
                for nc in range(4):
                    if nc == old:
                        continue
                    colors[ei] = nc
                    if distinct_count(eidx, colors) >= 5 and valid_move(ei, nc, colors, skip_k4=vk):
                        fixed = True
                        break
                    colors[ei] = old
                if fixed:
                    break
            if not fixed:
                # perturb to escape
                ei = rng.randrange(num_edges)
                nc = rng.randrange(4)
                old = colors[ei]
                colors[ei] = nc
                if not valid_move(ei, nc, colors):
                    colors[ei] = old
                stuck += 1
                if stuck > 400:
                    break
            viol = count_violations(colors)
        if viol == 0:
            best = colors[:]
            break

    if best is None:
        # fallback: known valid 17-colour construction
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]

    # ---- Phase 2: polish down to 4 colours ----
    colors = best[:]
    bad4 = count_bad4(colors)
    rng2 = random.Random(777)

    def current_palette(cs):
        return sorted(set(cs))

    while time.time() < deadline - 3:
        pal = current_palette(colors)
        if len(pal) <= 4:
            break
        progressed = False
        # try to eliminate colours, smallest count first
        counts = {}
        for c in colors:
            counts[c] = counts.get(c, 0) + 1
        order = sorted(pal, key=lambda c: counts[c])
        for target in order:
            if time.time() > deadline - 3:
                break
            edges_with = [i for i in range(num_edges) if colors[i] == target]
            rng2.shuffle(edges_with)
            for ei in edges_with:
                if time.time() > deadline - 3:
                    break
                if colors[ei] != target:
                    continue
                old = colors[ei]
                # try moving to another colour that keeps validity
                cands = [c for c in pal if c != target]
                rng2.shuffle(cands)
                moved = False
                for nc in cands:
                    colors[ei] = nc
                    if valid_move(ei, nc, colors):
                        new_bad4 = count_bad4(colors)
                        if new_bad4 <= bad4:
                            bad4 = new_bad4
                            moved = True
                            break
                        colors[ei] = old
                    else:
                        colors[ei] = old
                if moved:
                    progressed = True
                    # palette may have shrunk
                    pal2 = current_palette(colors)
                    if len(pal2) < len(pal):
                        pal = pal2
                        if len(pal) <= 4:
                            break
                    # recompute counts/order after shrink
                    if len(pal2) != len(pal):
                        counts = {}
                        for c in colors:
                            counts[c] = counts.get(c, 0) + 1
                        order = sorted(pal, key=lambda c: counts[c])
                    break
            if len(pal) <= 4:
                break
        if len(pal) <= 4:
            break
        if not progressed:
            # gentle perturbation: nudge an edge to reduce bad4
            ei = rng2.randrange(num_edges)
            old = colors[ei]
            nc = rng2.randrange(0, 5)
            colors[ei] = nc
            if valid_move(ei, nc, colors):
                nb = count_bad4(colors)
                if nb < bad4:
                    bad4 = nb
                else:
                    colors[ei] = old
            else:
                colors[ei] = old

    # final validity guard
    if count_violations(colors) > 0:
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    return colors
