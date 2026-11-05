import math
import random
import time

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _construct():
    deadline = time.time() + 70.0
    MAXM = 128
    MODS = list(range(2, MAXM + 1))
    N = len(MODS)
    idx = {m: i for i, m in enumerate(MODS)}
    gcdmat = [[_gcd(MODS[i], MODS[j]) for j in range(N)] for i in range(N)]

    def feasible(residue_sets, i, r):
        g = gcdmat[i]
        for j in range(N):
            if j == i:
                continue
            gj = g[j]
            if gj > 1:
                S = residue_sets[j]
                if S is None:
                    continue
                if (r - r) % gj == 0:
                    continue
                need = (r - r) % gj
                for rj in S:
                    if (rj - r) % gj != 0:
                        return True
                return False
        return True

    def pick_order(chosen):
        avail = [i for i in range(N) if i not in chosen]
        if not avail:
            return avail
        def score(i):
            cnt = 0
            g = gcdmat[i]
            for j in chosen:
                if g[j] > 1:
                    cnt += 1
            return (cnt, -MODS[i])
        return sorted(avail, key=score, reverse=True)

    def backtrack(chosen, residue_sets, chosen_res):
        order = pick_order(chosen)
        if not order:
            return chosen_res
        i = order[0]
        m = MODS[i]
        g = gcdmat[i]
        cands = [r for r in range(m) if feasible(residue_sets, i, r)]
        if not cands:
            return None
        random.shuffle(cands)
        for r in cands:
            new_sets = residue_sets + [None] * N
            new_sets[i] = {r}
            new_chosen = set(chosen)
            new_chosen.add(i)
            res = backtrack(new_chosen, new_sets, chosen_res + [[r, m]])
            if res is not None:
                return res
        return None

    def local_search(chosen_res):
        chosen = set(idx[mod] for _, mod in chosen_res)
        best = list(chosen_res)
        improved = True
        while improved:
            improved = False
            cur = set(idx[mod] for _, mod in best)
            for i in range(len(best)):
                r_i, m_i = best[i]
                i_idx = idx[m_i]
                for j in range(len(best)):
                    if j == i:
                        continue
                    r_j, m_j = best[j]
                    j_idx = idx[m_j]
                    for r_new in range(m_i):
                        if r_new == r_i:
                            continue
                        if not feasible(best, i_idx, r_new):
                            continue
                        test = list(best)
                        test[i] = [r_new, m_i]
                        for k in range(len(test)):
                            if k == i:
                                continue
                            r_k, m_k = test[k]
                            k_idx = idx[m_k]
                            g = gcdmat[i_idx][k_idx]
                            if g > 1 and (r_k - r_new) % g == 0:
                                break
                        else:
                            best = test
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best

    best = None
    for attempt in range(20000):
        if time.time() > deadline:
            break
        random.shuffle(MODS)
        chosen = set()
        sets = [None] * N
        chosen_res = []
        for i in range(N):
            m = MODS[i]
            g = gcdmat[i]
            cands = [r for r in range(m) if feasible(sets, i, r)]
            if not cands:
                continue
            r = random.choice(cands)
            chosen.add(i)
            sets[i] = {r}
            chosen_res.append([r, m])
        if len(chosen_res) > 1:
            res = local_search(chosen_res)
            if best is None or len(res) > len(best):
                best = res
                if len(best) >= N:
                    break
    if best is None:
        best = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    return best

def construct():
    return _construct()
