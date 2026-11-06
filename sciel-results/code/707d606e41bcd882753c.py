import time

_DEADLINE = 85.0
_T0 = time.time()

def _smooth_upto(limit):
    out = []
    for x in range(1, limit + 1):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            out.append(x)
    return out

def _search(n, smooth, kmax, budget):
    cand = [x for x in smooth if x <= n]
    m = len(cand)
    # suffix min/max sums of `need` from position idx
    # precompute prefix sums
    pref = [0] * (m + 1)
    for i in range(m):
        pref[i + 1] = pref[i] + cand[i]
    def minsum(idx, need):
        if m - idx < need:
            return None
        return pref[idx + need] - pref[idx]
    def maxsum(idx, need):
        if m - idx < need:
            return None
        return pref[m] - pref[m - need]
    for k in range(1, kmax + 1):
        if time.time() > budget:
            break
        res = []
        def bt(idx, chosen, remaining):
            if len(chosen) == k:
                if remaining == 0:
                    res.append(list(chosen))
                    return True
                return False
            need = k - len(chosen)
            ms = minsum(idx, need)
            if ms is None or ms > remaining:
                return False
            xs = maxsum(idx, need)
            if xs is None or xs < remaining:
                return False
            for i in range(idx, m):
                v = cand[i]
                if v > remaining:
                    break
                ok = True
                for c in chosen:
                    a, b = (c, v) if c < v else (v, c)
                    if b % a == 0:
                        ok = False
                        break
                if not ok:
                    continue
                chosen.append(v)
                if bt(i + 1, chosen, remaining - v):
                    chosen.pop()
                    return True
                chosen.pop()
            return False
        if bt(0, [], n):
            return res[0]
    return None

def _fallback(n, smooth):
    if n in smooth:
        return [n]
    cand = [x for x in smooth if x < n]
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            if cand[i] + cand[j] == n and cand[j] % cand[i] != 0:
                return [cand[i], cand[j]]
    for i in range(len(cand)):
        a = cand[i]
        for j in range(i + 1, len(cand)):
            b = cand[j]
            if b <= a:
                continue
            c = n - a - b
            if c <= b:
                continue
            if c in smooth and c % a != 0 and c % b != 0 and b % a != 0:
                return [a, b, c]
    return [n]

def construct():
    smooth = _smooth_upto(128)
    answers = []
    for n in range(1, 129):
        if time.time() > _DEADLINE:
            answers.append(_fallback(n, smooth))
            continue
        budget = min(_DEADLINE, time.time() + 0.5)
        r = _search(n, smooth, kmax=8, budget=budget)
        if r is None:
            r = _fallback(n, smooth)
        answers.append(r)
    return answers
