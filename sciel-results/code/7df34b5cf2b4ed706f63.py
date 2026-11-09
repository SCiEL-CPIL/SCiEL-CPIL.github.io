import time
import random

_DEADLINE = 80.0  # safety margin before 90s kill

def _square_nonzero_count(terms):
    d = {}
    n = len(terms)
    for i in range(n):
        pi, ci = terms[i]
        for j in range(i, n):
            pj, cj = terms[j]
            deg = pi + pj
            val = ci * cj
            if i != j:
                val *= 2
            d[deg] = d.get(deg, 0) + val
    cnt = 0
    for v in d.values():
        if v != 0:
            cnt += 1
    return cnt

def _to_coeff_list(terms):
    maxd = max(p for p, _ in terms)
    if maxd > 40:
        return None
    lst = [0] * (maxd + 1)
    for p, c in terms:
        lst[p] = c
    if lst[-1] == 0:
        return None
    nz = sum(1 for x in lst if x != 0)
    if nz != 12:
        return None
    for x in lst:
        if x < -64 or x > 64:
            return None
    return lst

def construct():
    start = time.time()
    rng = random.Random(12345)

    best_terms = None
    best_cnt = 10**9

    def consider(terms):
        nonlocal best_terms, best_cnt
        if len(terms) != 12:
            return
        if len(set(p for p, _ in terms)) != 12:
            return
        for p, c in terms:
            if p < 0 or p > 40 or c == 0 or c < -64 or c > 64:
                return
        cnt = _square_nonzero_count(terms)
        if cnt < best_cnt:
            best_cnt = cnt
            best_terms = list(terms)

    def random_terms():
        pos = rng.sample(range(0, 41), 12)
        terms = []
        for p in pos:
            c = rng.randint(1, 64)
            if rng.random() < 0.5:
                c = -c
            terms.append((p, c))
        return terms

    seed_terms = [(i, 1) for i in range(12)]
    consider(seed_terms)

    def local_search(terms, max_steps):
        nonlocal best_terms, best_cnt
        cur = list(terms)
        cur_cnt = _square_nonzero_count(cur)
        consider(cur)
        step = 0
        while step < max_steps:
            if time.time() - start > _DEADLINE:
                break
            step += 1
            idx = rng.randrange(12)
            p, c = cur[idx]
            new_terms = list(cur)
            r = rng.random()
            if r < 0.5:
                delta = rng.choice([-2, -1, 1, 2, -3, 3, -5, 5, -8, 8])
                nc = c + delta
                if nc == 0 or nc < -64 or nc > 64:
                    continue
                new_terms[idx] = (p, nc)
            else:
                delta = rng.choice([-2, -1, 1, 2, -3, 3, -5, 5])
                np_ = p + delta
                if np_ < 0 or np_ > 40:
                    continue
                # collision check
                collides = False
                for k in range(12):
                    if k != idx and cur[k][0] == np_:
                        collides = True
                        break
                if collides:
                    continue
                new_terms[idx] = (np_, c)
            new_cnt = _square_nonzero_count(new_terms)
            if new_cnt <= cur_cnt:
                cur = new_terms
                cur_cnt = new_cnt
                consider(cur)
            elif rng.random() < 0.05:
                cur = new_terms
                cur_cnt = new_cnt

    n_restarts = 0
    while time.time() - start < _DEADLINE:
        n_restarts += 1
        if n_restarts == 1:
            start_terms = seed_terms
        else:
            start_terms = random_terms()
        local_search(start_terms, 4000)

    if best_terms is None:
        return [1] * 12
    lst = _to_coeff_list(best_terms)
    if lst is None:
        return [1] * 12
    return lst
