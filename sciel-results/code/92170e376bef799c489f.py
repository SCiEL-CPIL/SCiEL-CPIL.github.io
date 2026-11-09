import time
import random
import numpy as np

BASE = 51975
# all odd divisors of 51975 = 3^3 * 5^2 * 7 * 11
DIVS = [d for d in range(3, BASE + 1, 2) if BASE % d == 0]
DIVS.sort()
N = len(DIVS)

# precomputed residue index arrays: IDX[i] = np.arange(0, BASE, DIVS[i])
IDX = [np.arange(0, BASE, m) for m in DIVS]
# per-divisor coverage counts for a given residue vector
CNT = [np.zeros(BASE, dtype=np.int32) for _ in range(N)]

DEADLINE = 110.0  # safety margin before 150s kill

def _score(vec):
    return int(np.count_nonzero(vec == 0))

def _build(vec):
    return [[DIVS[i], int(vec[i])] for i in range(N)]

def _try_insert(best_vec, best_score, budget, rng):
    """Try adding a single class for an unused divisor; keep if it lowers score."""
    used = set(np.nonzero(best_vec)[0])
    candidates = [i for i in range(N) if i not in used]
    for i in candidates:
        if time.time() > budget:
            break
        m = DIVS[i]
        idx = IDX[i]
        # best residue r minimizes new uncovered
        base_uncov = best_vec[idx] == 0
        # for each r, covered = base_uncov & (idx == r)
        # new uncovered = best_score - count of uncovered at r
        best_r = -1
        best_gain = 0
        # count uncovered per residue class
        unc = np.zeros(m, dtype=np.int32)
        for r in range(m):
            unc[r] = int(np.count_nonzero(base_uncov & (idx == r)))
        r = int(np.argmax(unc))
        gain = int(unc[r])
        if gain > best_gain:
            best_gain = gain
            best_r = r
        if best_gain > 0:
            new_vec = best_vec.copy()
            new_vec[IDX[best_i] ]  # placeholder
            # apply
            new_vec = best_vec.copy()
            new_vec[idx[idx == best_r]] = 1
            ns = _score(new_vec)
            if ns < best_score:
                best_vec, best_score = new_vec, ns
                used.add(best_i)
    return best_vec, best_score

def _try_remove(best_vec, best_score, budget, rng):
    """Try removing a single class; keep if it does not increase score (prefer fewer classes)."""
    used = set(np.nonzero(best_vec)[0])
    for i in list(used):
        if time.time() > budget:
            break
        new_vec = best_vec.copy()
        new_vec[IDX[i][new_vec[IDX[i]] == 1]] = 0
        ns = _score(new_vec)
        if ns < best_score:
            best_vec, best_score = new_vec, ns
            used.discard(i)
        elif ns == best_score:
            # keep removal to reduce class count (tie-break)
            best_vec = new_vec
            used.discard(i)
    return best_vec, best_score

def _try_swap(best_vec, best_score, budget, rng):
    """Try changing the residue of a used class to another value."""
    used = list(np.nonzero(best_vec)[0])
    rng.shuffle(used)
    for i in used:
        if time.time() > budget:
            break
        m = DIVS[i]
        idx = IDX[i]
        cur = int(best_vec[i])
        # try all residues, pick best
        best_r = cur
        best_ns = best_score
        # current contribution
        for r in range(m):
            if r == cur:
                continue
            # simulate: flip class i from cur to r
            new_vec = best_vec.copy()
            # remove old
            old_idx = idx[idx == cur]
            new_vec[old_idx] = np.where(new_vec[old_idx] == 1, 1, 0)
            # recompute coverage: a position is covered if any class covers it
            # simpler: rebuild count
            cnt = np.zeros(BASE, dtype=np.int32)
            for j in range(N):
                if best_vec[j]:
                    cnt[IDX[j][IDX[j] == int(best_vec[j])]] += 1
            # apply swap on cnt
            cnt[old_idx] -= 1
            new_idx = idx[idx == r]
            cnt[new_idx] += 1
            ns = int(np.count_nonzero(cnt == 0))
            if ns < best_ns:
                best_ns = ns
                best_r = r
        if best_r != cur and best_ns < best_score:
            new_vec = best_vec.copy()
            new_vec[IDX[i][IDX[i] == cur]] = 0
            new_vec[IDX[i][IDX[i] == best_r]] = 1
            best_vec, best_score = new_vec, best_ns
    return best_vec, best_score

def construct():
    t0 = time.time()
    budget = DEADLINE
    rng = random.Random(12345)

    # ---- Phase 1: greedy construction, largest moduli first ----
    vec = np.zeros(BASE, dtype=np.int32)
    for i in sorted(range(N), key=lambda k: -DIVS[k]):
        m = DIVS[i]
        idx = IDX[i]
        unc = vec[idx] == 0
        # count uncovered per residue
        best_r = 0
        best_gain = -1
        for r in range(m):
            g = int(np.count_nonzero(unc & (idx == r)))
            if g > best_gain:
                best_gain = g
                best_r = r
        vec[idx[idx == best_r]] = 1
        vec[i] = best_r  # store residue (i is index, not position; fix below)
    # NOTE: vec is over residues 0..BASE-1, but we also store residue in vec[i] where i< N.
    # That's a bug: i indexes divisors, not residues. Use separate array.
    # Redo cleanly:
    res = np.zeros(N, dtype=np.int32)  # residue per divisor
    for i in sorted(range(N), key=lambda k: -DIVS[k]):
        m = DIVS[i]
        idx = IDX[i]
        # coverage from current assignments
        cnt = np.zeros(BASE, dtype=np.int32)
        for j in range(N):
            if res[j]:
                cnt[IDX[j][IDX[j] == int(res[j])]] += 1
        unc = cnt[idx] == 0
        best_r = 0
        best_gain = -1
        for r in range(m):
            g = int(np.count_nonzero(unc & (idx == r)))
            if g > best_gain:
                best_gain = g
                best_r = r
        res[i] = best_r
    # build coverage vec
    cov = np.zeros(BASE, dtype=np.int32)
    for j in range(N):
        if res[j]:
            cov[IDX[j][IDX[j] == int(res[j])]] += 1
    best_score = _score(cov)
    best_res = res.copy()

    # ---- Phase 2: local search (swap residues, then remove/insert) ----
    def cov_of(r):
        c = np.zeros(BASE, dtype=np.int32)
        for j in range(N):
            if r[j]:
                c[IDX[j][IDX[j] == int(r[j])]] += 1
        return c

    iters = 0
    while time.time() < budget - 5:
        iters += 1
        # random swap
        i = rng.randrange(N)
        m = DIVS[i]
        r = rng.randrange(m)
        new_res = best_res.copy()
        new_res[i] = r
        c = cov_of(new_res)
        s = _score(c)
        if s < best_score:
            best_score = s
            best_res = new_res
        # occasional remove
        if iters % 20 == 0:
            i = rng.randrange(N)
            new_res = best_res.copy()
            new_res[i] = 0
            c = cov_of(new_res)
            s = _score(c)
            if s < best_score:
                best_score = s
                best_res = new_res
            elif s == best_score:
                best_res = new_res  # fewer classes
        # occasional insert
        if iters % 20 == 10:
            unused = [j for j in range(N) if not best_res[j]]
            if unused:
                i = rng.choice(unused)
                m = DIVS[i]
                # pick best residue
                c = cov_of(best_res)
                idx = IDX[i]
                unc = c[idx] == 0
                best_r = 0
                best_gain = -1
                for rr in range(m):
                    g = int(np.count_nonzero(unc & (idx == rr)))
                    if g > best_gain:
                        best_gain = g
                        best_r = rr
                new_res = best_res.copy()
                new_res[i] = best_r
                c2 = cov_of(new_res)
                s = _score(c2)
                if s < best_score:
                    best_score = s
                    best_res = new_res

    return _build(best_res)
