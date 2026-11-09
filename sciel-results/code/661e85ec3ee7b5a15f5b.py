import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130

    def solve_n(n):
        triples = list(combinations(range(n), 3))
        nt = len(triples)
        tidx = {t: i for i, t in enumerate(triples)}

        # Precompute quad partners: for triple i, list of (d, j, k, l)
        # where {a,b,c,d} is a quad with triples i,j,k,l
        quad_partners = [[] for _ in range(nt)]
        for i, (a, b, c) in enumerate(triples):
            for d in range(n):
                if d in (a, b, c):
                    continue
                t1 = tuple(sorted((a, b, d)))
                t2 = tuple(sorted((a, c, d)))
                t3 = tuple(sorted((b, c, d)))
                if t1 in tidx and t2 in tidx and t3 in tidx:
                    quad_partners[i].append((d, tidx[t1], tidx[t2], tidx[t3]))

        # Precompute quint partners: for triple i, list of (d, e, other_tris_indices)
        quint_partners = [[] for _ in range(nt)]
        for i, (a, b, c) in enumerate(triples):
            others = [v for v in range(n) if v not in (a, b, c)]
            for pi in range(len(others)):
                for pj in range(pi + 1, len(others)):
                    d, e = others[pi], others[pj]
                    quint = tuple(sorted((a, b, c, d, e)))
                    all_tris = list(combinations(quint, 3))
                    if all(t in tidx for t in all_tris):
                        other_idx = [tidx[t] for t in all_tris if t != (a, b, c)]
                        quint_partners[i].append((d, e, other_idx))

        rng = np.random.default_rng(42)
        colors = rng.integers(0, 2, nt)

        # Initial violation count
        red_v = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[tidx[(a,b,c)]] == 0 and colors[tidx[(a,b,d)]] == 0 and
                    colors[tidx[(a,c,d)]] == 0 and colors[tidx[(b,c,d)]] == 0):
                red_v += 1
        blue_v = 0
        for quint in combinations(range(n), 5):
            ok = all(colors[tidx[t]] == 1 for t in combinations(quint, 3))
            if ok:
                blue_v += 1
        total = red_v + blue_v
        if total == 0:
            return colors

        best_total = total
        best_colors = colors.copy()
        t0 = time.time()
        max_time = min(deadline - start_time - 10, 45)

        while time.time() - t0 < max_time:
            i = int(rng.integers(nt))
            old = colors[i]
            new = 1 - old

            # Delta red
            d_red = 0
            for (d, j, k, l) in quad_partners[i]:
                others_red = (colors[j] == 0 and colors[k] == 0 and colors[l] == 0)
                if others_red:
                    if old == 0 and new == 1:
                        d_red -= 1
                    elif old == 1 and new == 0:
                        d_red += 1

            # Delta blue
            d_blue = 0
            for (d, e, oidx) in quint_partners[i]:
                others_blue = all(colors[t] == 1 for t in oidx)
                if others_blue:
                    if old == 1 and new == 0:
                        d_blue -= 1
                    elif old == 0 and new == 1:
                        d_blue += 1

            new_total = total + d_red + d_blue
            if new_total <= total:
                colors[i] = new
                total = new_total
                if total == 0:
                    return colors
                if total < best_total:
                    best_total = total
                    best_colors = colors.copy()
            elif rng.random() < 0.02:
                colors[i] = new
                total = new_total

        return best_colors if best_total == 0 else None

    for n in range(14, 8, -1):
        if time.time() > deadline - 25:
            break
        result = solve_n(n)
        if result is not None:
            return {'n': n, 'colors': result.tolist()}

    # Fallback: n=9 with modular coloring
    n = 9
    c = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': c}
