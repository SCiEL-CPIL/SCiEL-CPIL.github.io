import time
import random

# EVOLVE-BLOCK-START
# --- constants ---
N = 284
PRES = (1, 2, 5, 11)
PRES_SET = frozenset(PRES)
PRES_LIST = list(PRES)
MAX_SUM = 2 * N
DEADLINE = 112.0          # seconds; hard safety margin under the 150s kill
CANDS = tuple(range(1, N + 1))
CANDS_NO_PRE = tuple(c for c in CANDS if c not in PRES_SET)
CANDS_BY_DIST = tuple(sorted(CANDS_NO_PRE, key=lambda c: min(abs(c - p) for p in PRES)))

# --- helpers ---
def _pres_sums():
    s = set()
    for i in range(4):
        for j in range(i, 4):
            s.add(PRES_LIST[i] + PRES_LIST[j])
    return s

def _init_conflict():
    conf = [0] * (MAX_SUM + 1)
    for i in range(4):
        for j in range(i, 4):
            conf[PRES_LIST[i] + PRES_LIST[j]] += 1
    return conf

def _greedy(used, conf, pres_sums):
    a = list(PRES_LIST)
    used = set(used)
    for c in CANDS_NO_PRE:
        if c in used:
            continue
        ok = True
        for y in a:
            if conf[c + y] != 0:
                ok = False
                break
        if not ok or conf[2 * c] != 0:
            continue
        for y in a:
            conf[c + y] += 1
        conf[2 * c] += 1
        pres_sums.add(2 * c)
        used.add(c)
        a.append(c)
    return a

def _try_insert(a, used, conf, pres_sums, c):
    for y in a:
        if conf[c + y] != 0:
            return False
    if conf[2 * c] != 0:
        return False
    for y in a:
        conf[c + y] += 1
    conf[2 * c] += 1
    pres_sums.add(2 * c)
    used.add(c)
    a.append(c)
    return True

def _remove(a, used, conf, pres_sums, c):
    idx = a.index(c)
    a.pop(idx)
    for y in a:
        conf[c + y] -= 1
    conf[2 * c] -= 1
    pres_sums.discard(2 * c)
    used.discard(c)

def _local_search(a, used, conf, pres_sums):
    best = len(a)
    improved = True
    while improved:
        improved = False
        for c in CANDS_NO_PRE:
            if c in used:
                continue
            if _try_insert(a, used, conf, pres_sums, c):
                improved = True
                if len(a) > best:
                    best = len(a)
        for idx in range(len(a) - 1, 0, -1):
            c = a[idx]
            if c in PRES_SET:
                continue
            _remove(a, used, conf, pres_sums, c)
            for d in CANDS_NO_PRE:
                if d in used:
                    continue
                if _try_insert(a, used, conf, pres_sums, d):
                    improved = True
                    break
            if len(a) < best:
                _remove(a, used, conf, pres_sums, d)
    return best

def _validate(a):
    if any(x < 1 or x > N for x in a):
        return False
    if any(p not in a for p in PRES):
        return False
    if len(set(a)) != len(a):
        return False
    seen = set()
    for i in range(len(a)):
        for j in range(i, len(a)):
            s = a[i] + a[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def construct():
    t0 = time.time()
    best = None
    best_len = 0

    # --- Phase 1: deterministic greedy (instant) ---
    conf = _init_conflict()
    pres_sums = _pres_sums()
    a = _greedy(set(), conf, pres_sums)
    if _validate(a):
        best = a
        best_len = len(a)

    # --- Phase 2: randomized search within the same family ---
    rng = random.Random(20240517)
    while time.time() - t0 < DEADLINE:
        # random initial subset of non-prescribed elements
        k = rng.randint(1, 9)
        init = list(rng.sample(CANDS_NO_PRE, k))
        used = set(init)
        conf = _init_conflict()
        pres_sums = _pres_sums()
        a = list(PRES_LIST)
        ok = True
        for c in init:
            for y in a:
                if conf[c + y] != 0:
                    ok = False
                    break
            if not ok or conf[2 * c] != 0:
                ok = False
                break
            for y in a:
                conf[c + y] += 1
            conf[2 * c] += 1
            pres_sums.add(2 * c)
            a.append(c)
        if not ok:
            continue
        a = _greedy(used, conf, pres_sums)
        _local_search(a, used, conf, pres_sums)
        if _validate(a) and len(a) > best_len:
            best_len = len(a)
            best = a

    if best is None:
        # fallback: just the prescribed set (always valid, size 4)
        return list(PRES)
    return best
# EVOLVE-BLOCK-END
