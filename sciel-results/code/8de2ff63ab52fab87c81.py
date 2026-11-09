import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 80.0

    N = 128
    # all nonconstant 4-term progressions (0-indexed positions)
    progs = []
    for d in range(1, 43):
        for a in range(0, N - 3 * d):
            progs.append((a, a + d, a + 2 * d, a + 3 * d))
    P = len(progs)
    pos2prog = [[] for _ in range(N)]
    for pi, (a, b, c, dd) in enumerate(progs):
        for p in (a, b, c, dd):
            pos2prog[p].append(pi)

    def full_eval(col):
        bad = 0
        for (a, b, c, dd) in progs:
            if len({col[a], col[b], col[c], col[dd]}) < 3:
                bad += 1
        return bad, len(set(col))

    def valid(col):
        for (a, b, c, dd) in progs:
            if len({col[a], col[b], col[c], col[dd]}) < 3:
                return False
        return True

    def sa_one(m, palette, seed, end_time):
        random.seed(seed * 1000003 + 7)
        if seed % 3 == 0:
            col = [i % m for i in range(N)]
        elif seed % 3 == 1:
            col = [random.choice(palette) for _ in range(N)]
        else:
            col = [(i * 7) % m for i in range(N)]
            for i in range(N):
                if random.random() < 0.5:
                    col[i] = random.choice(palette)
        prog_dist = [0] * P
        bad = 0
        for pi, (a, b, c, dd) in enumerate(progs):
            d_ = len({col[a], col[b], col[c], col[dd]})
            prog_dist[pi] = d_
            if d_ < 3:
                bad += 1
        T = 3.0
        maxit = 250000
        for it in range(maxit):
            if bad == 0:
                break
            if time.time() > end_time - 0.3:
                break
            p = random.randrange(N)
            old = col[p]
            new = random.choice(palette)
            if new == old:
                continue
            delta = 0
            for pi in pos2prog[p]:
                a, b, c, dd = progs[pi]
                ca = new if a == p else col[a]
                cb = new if b == p else col[b]
                cc_ = new if c == p else col[c]
                cd = new if dd == p else col[dd]
                nd = len({ca, cb, cc_, cd})
                od = prog_dist[pi]
                if od < 3 and nd >= 3:
                    delta -= 1
                elif od >= 3 and nd < 3:
                    delta += 1
            if delta <= 0 or random.random() < math.exp(-delta / T):
                col[p] = new
                for pi in pos2prog[p]:
                    a, b, c, dd = progs[pi]
                    nd = len({col[a], col[b], col[c], col[dd]})
                    prog_dist[pi] = nd
                bad += delta
            T = max(0.02, T * 0.9995)
        return col

    def minimize_used(col):
        # greedy color merging to reduce distinct colors while keeping validity
        cnt = {}
        for v in col:
            cnt[v] = cnt.get(v, 0) + 1
        changed = True
        while changed:
            changed = False
            colors = list(cnt.keys())
            for A in colors:
                if A not in cnt:
                    continue
                for B in colors:
                    if B == A or B not in cnt:
                        continue
                    pos = [i for i in range(N) if col[i] == A]
                    for i in pos:
                        col[i] = B
                    if valid(col):
                        cnt[B] = cnt.get(B, 0) + cnt[A]
                        del cnt[A]
                        changed = True
                        break
                    else:
                        for i in pos:
                            col[i] = A
                if changed:
                    break
        return col, len(cnt)

    def search_m(m, tl):
        end = min(t0 + DEADLINE - 2.0, time.time() + tl)
        palette = list(range(m))
        best_local = None
        best_local_used = 99
        restart = 0
        while time.time() < end - 0.5:
            restart += 1
            col = sa_one(m, palette, restart, end)
            if col is None:
                continue
            bad, _ = full_eval(col)
            if bad == 0:
                col2, used2 = minimize_used(col[:])
                if used2 < best_local_used:
                    best_local_used = used2
                    best_local = col2[:]
                if best_local_used == 3:
                    break
        return best_local

    def greedy_seed():
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3
                        for d in range(1, (x - 1) // 3 + 1))
                        and all(any(colors[x - 1 - j * d] != color for j in (1, 2))
                                for d in range(1, (x - 1) // 2 + 1))):
                    colors.append(color)
                    break
        return colors

    # try palette sizes; 4 first for a quick valid fallback, then 3 to beat it
    order = [4, 3, 5, 6, 7, 8]
    per_m = {3: 16, 4: 22, 5: 16, 6: 10, 7: 8, 8: 8}
    best = None
    best_used = 99
    for m in order:
        if time.time() - t0 > DEADLINE - 3:
            break
        if best_used == 3:
            break
        remaining = DEADLINE - (time.time() - t0) - 3
        tl = max(1.0, min(per_m[m], remaining))
        res = search_m(m, tl)
        if res is not None:
            used = len(set(res))
            if used < best_used:
                best_used = used
                best = res[:]

    if best is None:
        # last resort: a quick m=8 search with whatever time remains
        if time.time() - t0 < DEADLINE - 2:
            res = search_m(8, max(1.0, DEADLINE - (time.time() - t0) - 2))
            if res is not None:
                best = res
    if best is None:
        best = greedy_seed()

    # final safety: ensure validity; if not, fall back to seed
    bad, _ = full_eval(best)
    if bad != 0:
        best = greedy_seed()
    return best
# EVOLVE-BLOCK-END
