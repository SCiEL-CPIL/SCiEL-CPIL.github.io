# EVOLVE-BLOCK-START
import time
import random
import bisect

def _diffs_set(s):
    out = set()
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            out.add(s[j] - si)
    return out

def _valid_sidon(s):
    d = _diffs_set(s)
    return len(d) == len(s) * (len(s) - 1) // 2

def _try_add(a, x):
    if x in a:
        return None
    new_diffs = set()
    for v in a:
        d = abs(v - x)
        if d in new_diffs:
            return None
        new_diffs.add(d)
    existing = _diffs_set(a)
    if new_diffs & existing:
        return None
    trial = a + [x]
    trial.sort()
    return trial

def _covers_all(s):
    return set(range(1, 31)).issubset(_diffs_set(s))

def _greedy_build(seed, deadline):
    random.seed(seed)
    a = [0]
    target = set(range(1, 31))
    existing_diffs = set()

    while target:
        if time.time() > deadline:
            break
        best_trial = None
        best_cov = 0
        best_x = None

        candidates = set()
        for d in target:
            for v in a:
                x1 = v + d
                if 0 < x1 <= 10**6:
                    candidates.add(x1)
                x2 = v - d
                if 0 < x2 <= 10**6:
                    candidates.add(x2)
        max_a = max(a)
        for x in range(1, min(max_a + 60, 400)):
            candidates.add(x)
        for _ in range(300):
            candidates.add(random.randint(1, max(80, 3 * max_a)))

        cand_list = sorted(c for c in candidates if c not in a)
        if len(cand_list) > 800:
            cand_list = cand_list[:800]

        for x in cand_list:
            if time.time() > deadline:
                break
            trial = _try_add(a, x)
            if trial is None:
                continue
            new_diffs = set(abs(v - x) for v in a)
            cov = len(new_diffs & target)
            if cov > best_cov:
                best_cov = cov
                best_x = x
                best_trial = trial
                if best_cov == len(target):
                    break
            elif cov == best_cov and best_x is not None and x < best_x:
                best_x = x
                best_trial = trial

        if best_trial is None:
            # fallback: scan upward
            y = max(a) + 1
            while y < 10**6 and time.time() < deadline:
                trial = _try_add(a, y)
                if trial is not None:
                    new_diffs = set(abs(v - y) for v in a)
                    if new_diffs & target:
                        best_trial = trial
                        break
                y += 1
            if best_trial is None:
                y = 2 * max(a) + 100
                while time.time() < deadline:
                    trial = _try_add(a, y)
                    if trial is not None:
                        best_trial = trial
                        break
                    y += 1
            if best_trial is None:
                break

        a = best_trial
        all_diffs = _diffs_set(a)
        target -= all_diffs

    return a

def _shrink_diameter(a, deadline):
    """After full coverage, reduce the largest element and remove redundancy."""
    a = sorted(a)
    if not _covers_all(a) or not _valid_sidon(a):
        return a

    def try_replace_remove(i, trial):
        # trial must be Sidon and cover all
        if not _valid_sidon(trial):
            return None
        if not _covers_all(trial):
            return None
        return trial

    # Phase 1: repeatedly try to lower the largest element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        largest_idx = len(a) - 1
        old_val = a[largest_idx]
        # Try smaller values for the largest element
        # Candidate positions: values that create needed differences
        without = a[:largest_idx]
        # Build candidate set: for each missing-relevant diff and each remaining element
        need = set(range(1, 31))
        cur_diffs = _diffs_set(without)
        still_needed = need - cur_diffs
        cands = set()
        for d in still_needed:
            for v in without:
                c1 = v + d
                if 0 < c1 < old_val:
                    cands.add(c1)
                c2 = v - d
                if 0 < c2 < old_val:
                    cands.add(c2)
        # also small neighborhood
        for x in range(1, min(old_val, max(without) + 60)):
            cands.add(x)
        cands = sorted(c for c in cands if c not in without)
        for x in cands:
            trial = without + [x]
            if len(set(trial)) != len(trial):
                continue
            if not _valid_sidon(trial):
                continue
            if not _covers_all(trial):
                continue
            a = sorted(trial)
            improved = True
            break

    # Phase 2: remove redundant elements (keep smallest possible)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(a) - 1, 0, -1):  # remove from largest down, keep 0
            without = a[:i] + a[i+1:]
            if len(without) < 2:
                continue
            if _valid_sidon(without) and _covers_all(without):
                a = without
                changed = True
                break

    return a

def _kick(a, seed, deadline):
    """Random perturbation: drop a few elements, rebuild, shrink."""
    random.seed(seed)
    if len(a) <= 4:
        return a
    k = random.randint(1, min(3, len(a) - 2))
    keep = {0}
    others = [x for x in a if x != 0]
    removed = random.sample(others, k)
    kept_others = [x for x in others if x not in removed]
    seed_set = [0] + kept_others
    # rebuild from this seed
    a2 = sorted(seed_set)
    target = set(range(1, 31)) - _diffs_set(a2)
    while target and time.time() < deadline:
        best_trial = None
        best_cov = 0
        best_x = None
        cands = set()
        for d in target:
            for v in a2:
                x1 = v + d
                if 0 < x1 <= 10**6:
                    cands.add(x1)
                x2 = v - d
                if 0 < x2 <= 10**6:
                    cands.add(x2)
        for _ in range(200):
            cands.add(random.randint(1, max(80, 3 * max(a2))))
        cands = sorted(c for c in cands if c not in a2)
        if len(cands) > 500:
            cands = cands[:500]
        for x in cands:
            if time.time() > deadline:
                break
            trial = _try_add(a2, x)
            if trial is None:
                continue
            new_diffs = set(abs(v - x) for v in a2)
            cov = len(new_diffs & target)
            if cov > best_cov:
                best_cov = cov
                best_x = x
                best_trial = trial
                if best_cov == len(target):
                    break
        if best_trial is None:
            break
        a2 = best_trial
        target -= _diffs_set(a2)
    if not _covers_all(a2):
        return a
    return _shrink_diameter(a2, deadline)

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    best = None
    best_max = 10**9

    def consider(a):
        nonlocal best, best_max
        if a is None:
            return
        if not _valid_sidon(a) or not _covers_all(a):
            return
        if len(a) > 32:
            return
        m = max(a)
        if m < best_max:
            best_max = m
            best = sorted(a)

    # Phase A: multiple greedy builds with different seeds, shrink each
    for seed in range(40):
        if time.time() > deadline - 20:
            break
        raw = _greedy_build(seed, deadline - 20)
        if not _covers_all(raw):
            continue
        sh = _shrink_diameter(raw, deadline - 20)
        consider(sh)

    # Phase B: kick + shrink to escape local optima
    if best is not None and time.time() < deadline - 10:
        for it in range(60):
            if time.time() > deadline - 10:
                break
            kicked = _kick(best, 100000 + it, deadline - 10)
            consider(kicked)

    if best is None:
        # last-ditch deterministic fallback
        a = [0]
        for d in range(1, 31):
            if d in _diffs_set(a):
                continue
            found = False
            for x in range(1, 5000):
                trial = _try_add(a, x)
                if trial is not None and d in set(abs(v - x) for v in a):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + d + 1
                while True:
                    trial = _try_add(a, y)
                    if trial is not None:
                        trial2 = _try_add(trial, y + d)
                        if trial2 is not None:
                            a = trial2
                            break
                    y += 1
        consider(_shrink_diameter(a, deadline))

    return best
# EVOLVE-BLOCK-END
