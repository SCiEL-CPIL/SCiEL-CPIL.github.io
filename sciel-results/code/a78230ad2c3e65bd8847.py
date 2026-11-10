import time
from fractions import Fraction

def _valid_set(s):
    """Check that no reciprocal of a selected element equals the sum of
    reciprocals of a subset of the OTHER selected elements."""
    n = len(s)
    for i in range(n):
        target = Fraction(1, s[i])
        acc = 0
        for j in range(n):
            if j == i:
                continue
            acc += Fraction(1, s[j])
            if acc == target:
                return False
            if acc > target:
                break
    return True

def construct():
    start = time.time()
    deadline = start + 84.0

    # Candidates ordered by descending reciprocal value (smallest index first).
    order = list(range(1, 25))
    vals = [Fraction(1, x) for x in order]
    n = len(order)

    best = [1]
    best_len = 1
    memo = set()

    def greedy_completion(idx, S, s):
        local = list(s)
        localS = set(S)
        for k in range(idx, n):
            v = vals[k]
            if v in localS:
                continue
            local.append(order[k])
            newS = set()
            for a in localS:
                b = a + v
                if b <= 1:
                    newS.add(b)
            localS |= newS
            if time.time() > deadline:
                break
        return local

    def search(idx, S, s):
        nonlocal best, best_len
        ub = len(s)
        for k in range(idx, n):
            if vals[k] not in S:
                ub += 1
        if ub <= best_len:
            return
        if time.time() > deadline:
            return
        if idx == n:
            if len(s) > best_len:
                best_len = len(s)
                best = list(s)
            return

        key = (idx, frozenset(S))
        if key in memo:
            return
        memo.add(key)

        if time.time() > deadline - 1.5:
            gset = greedy_completion(idx, S, s)
            if len(gset) > best_len:
                best_len = len(gset)
                best = list(gset)
            return

        x = order[idx]
        v = vals[idx]

        if v not in S:
            newS = set(S)
            for a in S:
                b = a + v
                if b <= 1:
                    newS.add(b)
            s.append(x)
            search(idx + 1, newS, s)
            s.pop()

        search(idx + 1, S, s)

    search(0, {Fraction(0, 1)}, [])

    if not best or not _valid_set(best):
        fb = []
        S = {Fraction(0, 1)}
        for x in range(24, 0, -1):
            v = Fraction(1, x)
            if v in S:
                continue
            fb.append(x)
            for a in S:
                b = a + v
                if b <= 1:
                    S.add(b)
        if fb and _valid_set(fb):
            best = fb
        else:
            best = [1]

    return best
