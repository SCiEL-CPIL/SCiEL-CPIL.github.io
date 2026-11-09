import time
import random
import numpy as np

N = 256
MAX_EACH = 64
TIME_LIMIT = 90.0
MARGIN = 8.0

def _make_sumset(elem):
    s = np.zeros(2 * N + 2, dtype=bool)
    for a in elem:
        s[a + a] = True
    for i in range(len(elem)):
        ai = elem[i]
        for j in range(i + 1, len(elem)):
            s[ai + elem[j]] = True
    return s

def _make_blocker(elem, other):
    # own nonzero diffs OR the other set's nonzero diffs (symmetric)
    d = np.zeros(2 * N + 1, dtype=bool)
    for i in range(len(elem)):
        ai = elem[i]
        for j in range(i + 1, len(elem)):
            d[abs(elem[j] - ai)] = True
    for i in range(len(other)):
        ai = other[i]
        for j in range(i + 1, len(other)):
            d[abs(other[j] - ai)] = True
    return d

def _add(elem, cand, sums, blockers, sums_other, blockers_other, N=N):
    x = cand
    if sums[x + x] or np.any(sums[elem + x]):
        return False
    d = np.abs(elem - x)
    if d.size and np.any(blockers[d]):
        return False
    # commit to elem side
    sums[x + x] = True
    sums[elem + x] = True
    if d.size:
        blockers[d] = True
    # propagate x's diffs into the other side's blocker view
    if len(elem):
        s = np.abs(x - elem)
        if s.size:
            blockers_other[s] = True
    elem.append(x)
    return True

def _remove(elem, sums, blockers, blockers_other, N=N):
    x = elem.pop()
    if len(elem):
        d = np.abs(elem - x)
        sums[elem + x] = False
        blockers[d] = False
        s = np.abs(x - elem)
        if s.size:
            blockers_other[s] = False
    else:
        sums[x + x] = False
    return x

def _feasible(elem, x, sums, blockers):
    if sums[x + x] or np.any(sums[elem + x]):
        return False
    d = np.abs(elem - x)
    if d.size and np.any(blockers[d]):
        return False
    return True

def _add2(elem, cand, sums, blockers, blockers_other, N=N):
    x = cand
    if sums[x + x] or np.any(sums[elem + x]):
        return False
    d = np.abs(elem - x)
    if d.size and np.any(blockers[d]):
        return False
    sums[x + x] = True
    sums[elem + x] = True
    if d.size:
        blockers[d] = True
        s = np.abs(x - elem)
        if s.size:
            blockers_other[s] = True
    elem.append(x)
    return True

def _greedy(order, A0, B0, deadline, deadline2):
    A = list(A0); B = list(B0)
    sumsA = _make_sumset(A); sumsB = _make_sumset(B)
    blockersA = _make_blocker(A, B); blockersB = _make_blocker(B, A)
    for x in order:
        if len(A) < MAX_EACH and _feasible(A, x, sumsA, blockersA):
            _add(A, x, sumsA, blockersA, sumsB, blockersB)
        if len(B) < MAX_EACH and _feasible(B, x, sumsB, blockersB):
            _add(B, x, sumsB, blockersB, sumsA, blockersA)
        if (A and B and not (len(A) < MAX_EACH and len(B) < MAX_EACH)):
            break
    return A, B

def _local_search(A, B, deadline, deadline2):
    A = list(A); B = list(B)
    sumsA = _make_sumset(A); sumsB = _make_sumset(B)
    blockersA = _make_blocker(A, B); blockersB = _make_blocker(B, A)
    best_size = len(A) + len(B)
    bestA = list(A); bestB = list(B)
    cand = np.array([x for x in range(1, N + 1) if x not in A and x not in B], dtype=int)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline2:
            break
        # move 1: single add
        for x in cand.tolist():
            if len(A) < MAX_EACH and _feasible(A, x, sumsA, blockersA):
                _add(A, x, sumsA, blockersA, sumsB, blockersB); cand = cand[cand != x]; improved = True
                break
            if len(B) < MAX_EACH and _feasible(B, x, sumsB, blockersB):
                _add(B, x, sumsB, blockersB, sumsA, blockersA); cand = cand[cand != x]; improved = True
                break
        if improved:
            sz = len(A) + len(B)
            if sz > best_size:
                best_size = sz; bestA = list(A); bestB = list(B)
            if best_size >= 2 * MAX_EACH:
                break
            continue
        # move 2: 2-for-1 in A (remove a blocker, add two)
        if time.time() > deadline2:
            break
        done = False
        if len(A) > 1 and len(cand) >= 2 and len(A) < MAX_EACH:
            a_elem = np.array(A, dtype=int)
            for idx in range(len(A)):
                if time.time() > deadline2:
                    break
                rem = _remove(A, sumsA, blockersA, blockersB)
                cA = cand[np.array([_feasible(A, x, sumsA, blockersA) for x in cand.tolist()])]
                if len(cA) >= 2 and len(cA) + 1 > len(A):
                    x1 = cA[0]; x2 = cA[1]
                    _add2(A, x1, sumsA, blockersA, sumsB, blockersB)
                    cA2 = cA[cA != x1]
                    _add2(A, x2, sumsA, blockersA, sumsB, blockersB)
                    cand = cand[(cand != x1) & (cand != x2)]
                    sz = len(A) + len(B)
                    if sz > best_size:
                        best_size = sz; bestA = list(A); bestB = list(B)
                        done = True
                    else:
                        # revert to previous A
                        A.clear(); A.extend(bestA)
                        sumsA = _make_sumset(A); blockersA = _make_blocker(A, B)
                        blockersB = _make_blocker(B, A)
                    if done:
                        break
                else:
                    _add(A, rem, sumsA, blockersA, sumsB, blockersB)
            if done:
                improved = True
                if best_size >= 2 * MAX_EACH:
                    break
                continue
        # move 2b: 2-for-1 in B
        if time.time() > deadline2:
            break
        if len(B) > 1 and len(cand) >= 2 and len(B) < MAX_EACH:
            for idx in range(len(B)):
                if time.time() > deadline2:
                    break
                rem = _remove(B, sumsB, blockersB, sumsA)
                cB = cand[np.array([_feasible(B, x, sumsB, blockersB) for x in cand.tolist()])]
                if len(cB) >= 2 and len(cB) + 1 > len(B):
                    x1 = cB[0]; x2 = cB[1]
                    _add2(B, x1, sumsB, blockersB, sumsA, blockersA)
                    cB2 = cB[cB != x1]
                    _add2(B, x2, sumsB, blockersB, sumsA, blockersA)
                    cand = cand[(cand != x1) & (cand != x2)]
                    sz = len(A) + len(B)
                    if sz > best_size:
                        best_size = sz; bestA = list(A); bestB = list(B)
                        improved = True
                    else:
                        B.clear(); B.extend(bestB)
                        sumsB = _make_sumset(B); blockersB = _make_blocker(B, A)
                        blockersA = _make_blocker(A, B)
                    if improved:
                        break
                else:
                    _add(B, rem, sumsB, blockersB, sumsA, blockersA)
            if best_size >= 2 * MAX_EACH:
                break
    A.clear(); A.extend(bestA)
    B.clear(); B.extend(bestB)
    return A, B

def _validate(A, B):
    def ok(elem):
        s = set()
        for i in range(len(elem)):
            for j in range(i, len(elem)):
                t = elem[i] + elem[j]
                if t in s:
                    return False
                s.add(t)
        return True
    if not A or not B or len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    if not (set(A) <= set(range(1, N + 1)) and set(B) <= set(range(1, N + 1))):
        return False
    if not ok(A) or not ok(B):
        return False
    dA = set(); dB = set()
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            dA.add(A[j] - A[i]); dA.add(A[i] - A[j])
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            dB.add(B[j] - B[i]); dB.add(B[i] - B[j])
    return not (dA & dB)

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT - MARGIN
    deadline2 = t0 + TIME_LIMIT - 3.0

    # deterministic base (always valid)
    baseA, baseB = _greedy(list(range(1, N + 1)), [], [], deadline, deadline2)
    baseA, baseB = _local_search(baseA, baseB, deadline, deadline2)
    if not _validate(baseA, baseB):
        baseA, baseB = [1], [2]
    best_A, best_B = baseA, baseB
    best_size = len(baseA) + len(baseB)

    rng = random.Random(123456789)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        startA = list(best_A); startB = list(best_B)
        if rng.random() < 0.25:
            startA = []; startB = []
        else:
            for k in range(rng.randint(1, min(4, max(len(startA), 1)))):
                if startA:
                    startA.pop(rng.randrange(len(startA)))
            for k in range(rng.randint(1, min(4, max(len(startB), 1)))):
                if startB:
                    startB.pop(rng.randrange(len(startB)))
        if rng.random() < 0.5:
            order = list(range(1, N + 1)); rng.shuffle(order)
        else:
            order = list(range(1, N + 1, 2)) + list(range(2, N + 1, 2))
            rng.shuffle(order)
        A, B = _greedy(order, startA, startB, deadline, deadline2)
        A, B = _local_search(A, B, deadline, deadline2)
        if _validate(A, B):
            sz = len(A) + len(B)
            if sz > best_size:
                best_size = sz
                best_A, best_B = list(A), list(B)
                if best_size >= 2 * MAX_EACH:
                    break
    if not _validate(best_A, best_B):
        return [baseA, baseB]
    return [best_A, best_B]

if __name__ == "__main__":
    A, B = construct()
    print(len(A), len(B), len(A) + len(B))
    print(_validate(A, B))
