import time
import random

N = 256
DEADLINE = 52.0

def _sums_of(S):
    sums = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i, n):
            sums.add(a + S[j])
    return sums

def _blocked_set(S, sums):
    blocked = set()
    for s in sums:
        for a in S:
            x = s - a
            if 1 <= x <= N:
                blocked.add(x)
        if s % 2 == 0:
            x = s // 2
            if 1 <= x <= N:
                blocked.add(x)
    return blocked

def _is_sidon(S):
    sums = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i, n):
            v = a + S[j]
            if v in sums:
                return False
            sums.add(v)
    return True

def _greedy_to_maximal(S, seed, deadline):
    random.seed(seed)
    S = list(S)
    sums = _sums_of(S)
    Sset = set(S)
    blocked = _blocked_set(S, sums)
    while True:
        if time.time() > deadline:
            return S
        unblocked = [x for x in range(1, N + 1) if x not in Sset and x not in blocked]
        if not unblocked:
            return S
        best_x = None
        best_gain = -1
        cand = unblocked[:]
        random.shuffle(cand)
        for x in cand:
            new_sums = set()
            ok = True
            for a in S:
                v = x + a
                if v in sums or v in new_sums:
                    ok = False
                    break
                new_sums.add(v)
            if not ok:
                continue
            v = 2 * x
            if v in sums or v in new_sums:
                continue
            new_sums.add(v)
            # newly blocked from new_sums only
            new_blocked = set()
            for s in new_sums:
                for a in S:
                    y = s - a
                    if 1 <= y <= N:
                        new_blocked.add(y)
                if s % 2 == 0:
                    y = s // 2
                    if 1 <= y <= N:
                        new_blocked.add(y)
            gain = 0
            for y in new_blocked:
                if y not in blocked and y not in Sset and y != x:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None:
            return S  # stuck, not maximal
        S.append(best_x)
        Sset.add(best_x)
        sums.add(2 * best_x)
        for a in S[:-1]:
            sums.add(best_x + a)
        blocked = _blocked_set(S, sums)

def _local_search_reduce(S, deadline):
    """Try to reduce size by removing an element and re-filling; keep if smaller and maximal."""
    best = list(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(best)):
            if time.time() > deadline:
                break
            trial = best[:idx] + best[idx+1:]
            if not _is_sidon(trial):
                continue
            # re-fill to maximal
            filled = _greedy_to_maximal(trial, random.randint(0, 10**9), deadline)
            if filled is None:
                continue
            if not _is_sidon(filled):
                continue
            # check maximal
            sums = _sums_of(filled)
            blocked = _blocked_set(filled, sums)
            Sset = set(filled)
            unblocked = [x for x in range(1, N + 1) if x not in Sset and x not in blocked]
            if unblocked:
                continue
            if len(filled) < len(best):
                best = filled
                improved = True
                break
    return best

def construct():
    deadline = time.time() + DEADLINE
    best = None
    best_size = N + 1

    # Strategy 1: random small starts + greedy
    attempt = 0
    while time.time() < deadline - 2.0:
        attempt += 1
        seed = attempt
        random.seed(seed)
        start = []
        sums = set()
        target = random.randint(3, 10)
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if len(start) >= target:
                break
            new_sums = set()
            ok = True
            for a in start:
                v = x + a
                if v in sums or v in new_sums:
                    ok = False
                    break
                new_sums.add(v)
            if ok:
                v = 2 * x
                if v in sums or v in new_sums:
                    ok = False
            if ok:
                start.append(x)
                sums.update(new_sums)
                sums.add(2 * x)
        if len(start) < 3:
            continue
        S = _greedy_to_maximal(start, seed, deadline)
        if S is None:
            continue
        if not _is_sidon(S):
            continue
        sums = _sums_of(S)
        blocked = _blocked_set(S, sums)
        Sset = set(S)
        unblocked = [x for x in range(1, N + 1) if x not in Sset and x not in blocked]
        if unblocked:
            continue
        # local search reduce
        S2 = _local_search_reduce(S, deadline)
        if len(S2) < len(S):
            S = S2
        if len(S) < best_size:
            best_size = len(S)
            best = S
            if best_size <= 6:
                break

    if best is None:
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
    return best
