import time
import numpy as np

def construct(q, n, radius):
    start = time.time()
    DEADLINE = 118.0  # wall-clock safety margin before the 120s kill

    def remaining():
        return DEADLINE - (time.time() - start)

    def digits_to_int(v):
        out = 0
        for x in v:
            out = out * q + x
        return out

    def int_to_digits(x):
        d = []
        for _ in range(n):
            d.append(x % q)
            x //= q
        return d

    def build_neigh():
        # neigh[center] = list of ints within Hamming distance radius of center
        neigh = np.empty(q ** n, dtype=object)
        for c in range(q ** n):
            digs = int_to_digits(c)
            arr = np.empty((n, q), dtype=np.int64)
            for i in range(n):
                arr[i, :] = np.arange(q)
                arr[i, digs[i]] = digs[i]
            mask = np.zeros((n, q), dtype=bool)
            mask[:, digs] = True
            # enumerate subsets of size <= radius
            subs = [[]]
            for _ in range(radius):
                subs = subs + [s + [i] for s in subs for i in range(n) if i not in s]
            vals = set()
            for s in subs:
                if not s:
                    vals.add(c)
                    continue
                idx = np.array(s)
                cur = arr[idx, digs[idx]]
                for i in range(len(s)):
                    for a in range(q):
                        if a == digs[s[i]]:
                            continue
                        cur[i] = a
                        vals.add(int(np.prod(cur * np.array([q ** (n - 1 - i) for i in range(len(s))])) +
                                     int(np.prod((np.array([q ** (n - 1 - i) for i in range(len(s))]) * 0)))) )
            neigh[c] = list(vals)
        return neigh

    # Generic, robust coverage via digit arrays (no precomputed neighbor table for large spaces)
    def coverage_set(codes):
        codes = np.array(codes, dtype=np.int64)
        covered = np.zeros(q ** n, dtype=bool)
        for c in codes:
            digs = int_to_digits(c)
            arr = np.empty((n, q), dtype=np.int64)
            for i in range(n):
                arr[i, :] = np.arange(q)
                arr[i, digs[i]] = digs[i]
            subs = [[]]
            for _ in range(radius):
                subs = subs + [s + [i] for s in subs for i in range(n) if i not in s]
            for s in subs:
                if not s:
                    covered[c] = True
                    continue
                idx = np.array(s)
                cur = arr[idx, digs[idx]]
                base = c
                # compute the integer value for each combination
                for combo in _itertools_product(range(q), repeat=len(s)):
                    val = base
                    for i in range(len(s)):
                        pos = s[i]
                        cur[i] = combo[i]
                        if combo[i] != digs[pos]:
                            val += (combo[i] - digs[pos]) * (q ** (n - 1 - pos))
                    covered[val] = True
        return covered

    def _itertools_product(it, repeat):
        import itertools
        return itertools.product(it, repeat=repeat)

    def full_covered(codes):
        # returns boolean array of all q**n words
        covered = np.zeros(q ** n, dtype=bool)
        for c in codes:
            digs = int_to_digits(c)
            arr = np.empty((n, q), dtype=np.int64)
            for i in range(n):
                arr[i, :] = np.arange(q)
                arr[i, digs[i]] = digs[i]
            subs = [[]]
            for _ in range(radius):
                subs = subs + [s + [i] for s in subs for i in range(n) if i not in s]
            for s in subs:
                if not s:
                    covered[c] = True
                    continue
                idx = np.array(s)
                cur = arr[idx, digs[idx]]
                for combo in _itertools_product(range(q), repeat=len(s)):
                    val = c
                    for i in range(len(s)):
                        pos = s[i]
                        cur[i] = combo[i]
                        if combo[i] != digs[pos]:
                            val += (combo[i] - digs[pos]) * (q ** (n - 1 - pos))
                    covered[val] = True
        return covered

    # ---- Greedy seed: pick centers that cover the most still-uncovered words ----
    covered = np.zeros(q ** n, dtype=bool)
    codes = []
    # Precompute for each center the set of words it covers (as a list of ints)
    center_cov = {}
    for c in range(q ** n):
        digs = int_to_digits(c)
        arr = np.empty((n, q), dtype=np.int64)
        for i in range(n):
            arr[i, :] = np.arange(q)
            arr[i, digs[i]] = digs[i]
        subs = [[]]
        for _ in range(radius):
            subs = subs + [s + [i] for s in subs for i in range(n) if i not in s]
        lst = []
        for s in subs:
            if not s:
                lst.append(c)
                continue
            idx = np.array(s)
            cur = arr[idx, digs[idx]]
            for combo in _itertools_product(range(q), repeat=len(s)):
                val = c
                for i in range(len(s)):
                    pos = s[i]
                    cur[i] = combo[i]
                    if combo[i] != digs[pos]:
                        val += (combo[i] - digs[pos]) * (q ** (n - 1 - pos))
                lst.append(val)
        center_cov[c] = lst

    while not covered.all() and remaining() > 30.0:
        best_c = -1
        best_gain = -1
        for c in range(q ** n):
            gain = 0
            for w in center_cov[c]:
                if not covered[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_gain <= 0:
            break
        codes.append(best_c)
        for w in center_cov[best_c]:
            covered[w] = True

    # ---- Local search: delete redundant, then swap to improve coverage ----
    # Represent codes as a set of ints
    code_set = set(codes)

    def recompute_covered(cs):
        covered = np.zeros(q ** n, dtype=bool)
        for c in cs:
            for w in center_cov[c]:
                covered[w] = True
        return covered

    # Delete phase: remove any code whose removal keeps full coverage
    if covered.all():
        for c in list(code_set):
            if remaining() < 5.0:
                break
            trial = list(code_set)
            trial.remove(c)
            cv = recompute_covered(trial)
            if cv.all():
                code_set.remove(c)
                covered = cv
            if remaining() < 10.0:
                break

    # Swap phase: try replacing one code with another to keep/improve coverage
    all_codes = list(range(q ** n))
    improved = True
    while improved and remaining() > 15.0:
        improved = False
        cur_covered = recompute_covered(code_set)
        # find uncovered count
        uncov = np.where(~cur_covered)[0]
        if len(uncov) == 0:
            # try to reduce: for each code, see if swapping in a better one reduces size
            # we can't reduce below current if covered; instead try swaps that keep coverage
            # and allow deletion later. For now, try to find a swap that enables a deletion.
            pass
        for c in list(code_set):
            if remaining() < 8.0:
                break
            # candidate replacements: codes that cover many currently-uncovered words
            # only useful if there are uncovered words
            if len(uncov) == 0:
                continue
            best_new = -1
            best_gain = -1
            for cand in all_codes:
                if cand in code_set:
                    continue
                gain = 0
                for w in center_cov[cand]:
                    if not cur_covered[w]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_new = cand
            if best_new >= 0 and best_gain > 0:
                # check coverage after swap
                trial = set(code_set)
                trial.remove(c)
                trial.add(best_new)
                cv = recompute_covered(trial)
                if cv.sum() > cur_covered.sum():
                    code_set = trial
                    cur_covered = cv
                    improved = True
                    uncov = np.where(~cv)[0]

    # Final delete sweep
    if cur_covered.all():
        for c in list(code_set):
            if remaining() < 5.0:
                break
            trial = list(code_set)
            trial.remove(c)
            cv = recompute_covered(trial)
            if cv.all():
                code_set.remove(c)
            if remaining() < 10.0:
                break

    result = [int_to_digits(c) for c in code_set]
    # Ensure distinct
    seen = set()
    out = []
    for r in result:
        t = tuple(r)
        if t not in seen:
            seen.add(t)
            out.append(r)
    return out
