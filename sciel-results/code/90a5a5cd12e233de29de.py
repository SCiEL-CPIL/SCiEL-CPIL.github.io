"""Closest strict underpacking of one by unit fractions 1/2..1/48.
Scale by D=lcm(2..48) -> integer weights; maximize W<D via local search + restarts."""
# EVOLVE-BLOCK-START
def construct():
    import time, math, random
    start = time.time()
    DEADLINE = start + 55.0

    # D = lcm(2..48)
    D = 1
    for d in range(2, 49):
        D = D * d // math.gcd(D, d)

    denoms = list(range(2, 49))
    w = [D // d for d in denoms]
    n = len(w)
    all_indices = list(range(n))

    best_W = -1
    best_sel = None

    def apply_and_track(selected, total):
        nonlocal best_W, best_sel
        if total > best_W:
            best_W = total
            best_sel = set(selected)

    def try_improve(selected, total, rng):
        nonlocal best_W, best_sel
        wloc = w
        nloc = n
        Dloc = D
        while time.time() < DEADLINE:
            if total > best_W:
                best_W = total
                best_sel = set(selected)
            cur_def = Dloc - total
            best_move = None
            best_d = cur_def
            # single add
            for i in range(nloc):
                if i not in selected:
                    nt = total + wloc[i]
                    if nt < Dloc:
                        d = Dloc - nt
                        if d < best_d:
                            best_d = d
                            best_move = (0, i)
            sel = list(selected)
            # 1-out 1-in
            for r in sel:
                base = total - wloc[r]
                for a in range(nloc):
                    if a not in selected:
                        nt = base + wloc[a]
                        if nt < Dloc:
                            d = Dloc - nt
                            if d < best_d:
                                best_d = d
                                best_move = (1, r, a)
            # 1-out 2-in
            for r in sel:
                base = total - wloc[r]
                for a in range(nloc):
                    if a in selected:
                        continue
                    t1 = base + wloc[a]
                    if t1 >= Dloc:
                        continue
                    for b in range(a + 1, nloc):
                        if b in selected:
                            continue
                        nt = t1 + wloc[b]
                        if nt < Dloc:
                            d = Dloc - nt
                            if d < best_d:
                                best_d = d
                                best_move = (2, r, a, b)
            # 2-out 1-in
            L = len(sel)
            for i in range(L):
                r1 = sel[i]
                for j in range(i + 1, L):
                    r2 = sel[j]
                    base = total - wloc[r1] - wloc[r2]
                    for a in range(nloc):
                        if a in selected:
                            continue
                        nt = base + wloc[a]
                        if nt < Dloc:
                            d = Dloc - nt
                            if d < best_d:
                                best_d = d
                                best_move = (3, r1, r2, a)
            if best_move is None:
                # random kick to escape local optimum (feasible, may worsen)
                kicked = False
                for _ in range(3):
                    if len(selected) >= 1:
                        r = rng.choice(sel)
                        cands = [i for i in range(nloc) if i not in selected]
                        if cands:
                            a = rng.choice(cands)
                            nt = total - wloc[r] + wloc[a]
                            if nt < Dloc:
                                selected.remove(r)
                                selected.add(a)
                                total = nt
                                kicked = True
                                break
                if not kicked:
                    break
                continue
            m = best_move[0]
            if m == 0:
                i = best_move[1]
                selected.add(i); total += wloc[i]
            elif m == 1:
                r, a = best_move[1], best_move[2]
                selected.remove(r); selected.add(a)
                total = total - wloc[r] + wloc[a]
            elif m == 2:
                r, a, b = best_move[1], best_move[2], best_move[3]
                selected.remove(r); selected.add(a); selected.add(b)
                total = total - wloc[r] + wloc[a] + wloc[b]
            else:
                r1, r2, a = best_move[1], best_move[2], best_move[3]
                selected.remove(r1); selected.remove(r2); selected.add(a)
                total = total - wloc[r1] - wloc[r2] + wloc[a]
        return selected, total

    def init_bestfit(rng):
        rem = list(all_indices)
        rng.shuffle(rem)
        remset = set(rem)
        total = 0
        selected = set()
        wloc = w; Dloc = D
        while remset:
            best = None; bd = None
            for i in remset:
                nt = total + wloc[i]
                if nt < Dloc:
                    d = Dloc - nt
                    if bd is None or d < bd:
                        bd = d; best = i
            if best is None:
                break
            selected.add(best); total += wloc[best]; remset.discard(best)
        return selected, total

    def init_remove(rng):
        selected = set(all_indices)
        total = sum(w)
        order = list(all_indices)
        rng.shuffle(order)
        wloc = w; Dloc = D
        for i in order:
            if total < Dloc:
                break
            if i in selected:
                selected.discard(i)
                total -= wloc[i]
        return selected, total

    rng = random.Random(20240601)
    # deterministic strong start
    sel, tot = init_bestfit(random.Random(1))
    if not sel:
        sel = {0}; tot = w[0]
    try_improve(sel, tot, rng)

    # diverse random restarts
    while time.time() < DEADLINE:
        r = random.Random()
        if rng.random() < 0.5:
            sel, tot = init_bestfit(r)
        else:
            sel, tot = init_remove(r)
        if not sel:
            sel = {0}; tot = w[0]
        try_improve(sel, tot, rng)

    if best_sel is None or not best_sel:
        return [0]
    return sorted(best_sel)
# EVOLVE-BLOCK-END
