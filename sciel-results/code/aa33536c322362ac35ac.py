import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 83.0

    MOD_MAX = 128
    MODULI = list(range(2, MOD_MAX + 1))

    # Precompute gcd table
    M = MOD_MAX + 1
    gcd_table = [[0] * M for _ in range(M)]
    for i in range(2, M):
        for j in range(i, M):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def valid_residues(m, selected):
        out = []
        constraints = []
        for r_k, m_k in selected:
            g = gcd_table[m][m_k]
            if g == 1:
                return []
            constraints.append((g, r_k % g))
        if not constraints:
            return list(range(m))
        for r in range(m):
            ok = True
            for g, rk_mod in constraints:
                if r % g == rk_mod:
                    ok = False
                    break
            if ok:
                out.append(r)
        return out

    def greedy(order, rng=None):
        selected = []
        for m in order:
            vr = valid_residues(m, selected)
            if vr:
                r = rng.choice(vr) if rng else vr[0]
                selected.append((r, m))
        return [[r, m] for r, m in selected]

    def most_constrained_greedy(rng=None):
        remaining = set(MODULI)
        selected = []
        while remaining:
            if time.time() > deadline:
                break
            best_m = None
            best_vr = None
            best_count = 10**9
            for m in list(remaining):
                vr = valid_residues(m, selected)
                c = len(vr)
                if c == 0:
                    remaining.discard(m)
                    continue
                if c < best_count:
                    best_count = c
                    best_m = m
                    best_vr = vr
                    if c == 1:
                        break
            if best_m is None:
                break
            r = rng.choice(best_vr) if rng else best_vr[0]
            selected.append((r, best_m))
            remaining.discard(best_m)
        return [[r, m] for r, m in selected]

    def local_search(sol, max_rounds=100, rng=None):
        sol = [list(x) for x in sol]
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False

            # 1: add one
            sel_mods = set(m for _, m in sol)
            for m in MODULI:
                if m in sel_mods:
                    continue
                vr = valid_residues(m, sol)
                if vr:
                    r = rng.choice(vr) if rng else vr[0]
                    sol.append([r, m])
                    sel_mods.add(m)
                    improved = True
                    break
            if improved:
                continue

            # 2: 1-for-1
            for i in range(len(sol)):
                if time.time() > deadline:
                    return sol
                base = sol[:i] + sol[i+1:]
                base_mods = set(m for _, m in base)
                for m in MODULI:
                    if m in base_mods:
                        continue
                    vr = valid_residues(m, base)
                    if vr:
                        r = rng.choice(vr) if rng else vr[0]
                        base.append([r, m])
                        base_mods.add(m)
                if len(base) > len(sol):
                    sol = base
                    improved = True
                    break
            if improved:
                continue

            # 3: 2-for-1
            n = len(sol)
            for i in range(n):
                if time.time() > deadline:
                    return sol
                for j in range(i + 1, n):
                    base = sol[:i] + sol[i+1:j] + sol[j+1:]
                    base_mods = set(m for _, m in base)
                    for m in MODULI:
                        if m in base_mods:
                            continue
                        vr = valid_residues(m, base)
                        if vr:
                            r = rng.choice(vr) if rng else vr[0]
                            base.append([r, m])
                            base_mods.add(m)
                    if len(base) > len(sol):
                        sol = base
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue

            # 4: re-optimize residue
            for i in range(len(sol)):
                if time.time() > deadline:
                    return sol
                r_old, m_old = sol[i]
                others = [x for idx, x in enumerate(sol) if idx != i]
                vr = valid_residues(m_old, others)
                if len(vr) <= 1:
                    continue
                for r_new in vr:
                    if r_new == r_old:
                        continue
                    trial = [x[:] for x in sol]
                    trial[i] = [r_new, m_old]
                    trial_mods = set(m for _, m in trial)
                    for m in MODULI:
                        if m in trial_mods:
                            continue
                        vr2 = valid_residues(m, trial)
                        if vr2:
                            r_add = rng.choice(vr2) if rng else vr2[0]
                            trial.append([r_add, m])
                            trial_mods.add(m)
                    if len(trial) > len(sol):
                        sol = trial
                        improved = True
                        break
                if improved:
                    break

        return sol

    best = []
    best_len = 0

    # Prime-based init
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline - 5:
            break
        multiples = [m for m in MODULI if m % p == 0]
        k = min(len(multiples), p)
        sol = [[i % p, multiples[i]] for i in range(k)]
        if len(sol) > best_len:
            best = sol
            best_len = len(sol)

    # Greedy with orders
    def num_div(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c

    for order in [list(range(2, MOD_MAX + 1)),
                  list(range(MOD_MAX, 1, -1)),
                  sorted(MODULI, key=lambda x: -num_div(x))]:
        if time.time() > deadline - 10:
            break
        sol = greedy(order)
        if len(sol) > best_len:
            best = sol
            best_len = len(sol)

    sol = most_constrained_greedy()
    if len(sol) > best_len:
        best = sol
        best_len = len(sol)

    best = local_search(best, max_rounds=200)
    best_len = len(best)

    for it in range(600):
        if time.time() > deadline - 8:
            break
        rng = random.Random(100 + it)
        o = MODULI[:]
        rng.shuffle(o)
        sol = greedy(o, rng=rng)
        sol = local_search(sol, max_rounds=40, rng=rng)
        if len(sol) > best_len:
            best = sol
            best_len = len(sol)

    for it in range(150):
        if time.time() > deadline - 5:
            break
        rng = random.Random(5000 + it)
        sol = most_constrained_greedy(rng=rng)
        sol = local_search(sol, max_rounds=60, rng=rng)
        if len(sol) > best_len:
            best = sol
            best_len = len(sol)

    if time.time() < deadline - 3:
        best = local_search(best, max_rounds=300)

    result = []
    for r, m in best:
        r = r % m
        if 0 <= r < m:
            result.append([r, m])
    if not result:
        result = [[0, 2]]
    return result
