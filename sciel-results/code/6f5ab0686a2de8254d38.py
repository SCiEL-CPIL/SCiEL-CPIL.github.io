import random
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before the 150 s kill

    def _check(S):
        """Return (is_sidon, covered) for a sorted list S."""
        n = len(S)
        diffs = set()
        ok = True
        for i in range(n):
            si = S[i]
            for j in range(i + 1, n):
                d = S[j] - si
                if d in diffs:
                    ok = False
                    break
                diffs.add(d)
            if not ok:
                break
        if ok:
            # fast coverage test: every d in 1..30 must be present
            for d in range(1, 31):
                if d not in diffs:
                    ok = False
                    break
        return ok, diffs

    def _valid(S):
        ok, _ = _check(S)
        return ok

    def _covers(S):
        diffs = set(y - x for i, x in enumerate(S) for y in S[i + 1:])
        return all(d in diffs for d in range(1, 31))

    def _is_sidon(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            si = S[i]
            for j in range(i + 1, n):
                d = S[j] - si
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # ---------- Phase 1: fast randomized construction ----------
    def greedy_construct():
        S = [0]
        covered = {0}
        for _ in range(60):
            if covered.issuperset(range(1, 31)):
                break
            missing = [d for d in range(1, 31) if d not in covered]
            if not missing:
                break
            d = random.choice(missing)
            # candidate elements that realize d against some existing element
            cands = set()
            for x in S:
                cands.add(x + d)
                cands.add(x - d)
            cands = [c for c in cands if c >= 0 and c not in S]
            cands.sort()
            placed = None
            for c in cands:
                newS = S + [c]
                newS.sort()
                if _is_sidon(newS):
                    placed = c
                    break
            if placed is None:
                # add a large element to realize d, then a partner
                base = max(S) + d + 1
                placed = None
                for y in range(base, base + 4000):
                    trial = S + [y]
                    trial.sort()
                    if _is_sidon(trial):
                        placed = y
                        break
                if placed is None:
                    return None
            S.append(placed)
            S.sort()
            covered = set(y - x for i, x in enumerate(S) for y in S[i + 1:])
        if covered.issuperset(range(1, 31)):
            return S
        return None

    # ---------- Phase 2: local search to shrink the diameter ----------
    def local_search(S):
        S = sorted(S)
        best = S[:]
        best_max = max(S)
        n = len(S)
        # precompute for speed: current diff set
        while True:
            if time.time() - start_time > deadline - 5.0:
                break
            improved = False

            # (a) remove any element that is not needed for coverage
            for i in range(len(S)):
                trial = S[:i] + S[i + 1:]
                if len(trial) < 2:
                    continue
                if _is_sidon(trial) and _covers(trial):
                    if max(trial) < best_max:
                        S = trial
                        best = S[:]
                        best_max = max(S)
                        improved = True
                        break
            if improved:
                continue

            # (b) try to shrink a large element (skip 0)
            for i in range(len(S)):
                if S[i] == 0:
                    continue
                # try smaller values descending; stop after some budget
                lo = 1
                hi = S[i] - 1
                tried = 0
                for nv in range(hi, lo - 1, -1):
                    tried += 1
                    if tried > 60:
                        break
                    if nv in S:
                        continue
                    trial = S[:i] + [nv] + S[i + 1:]
                    trial.sort()
                    if _is_sidon(trial) and _covers(trial):
                        if max(trial) < best_max:
                            S = trial
                            best = S[:]
                            best_max = max(S)
                            improved = True
                        break
                if improved:
                    break

            if not improved:
                break
        return best

    # ---------- Phase 3: deterministic fallback (guaranteed valid) ----------
    def fallback():
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    best_set = None
    best_max = float('inf')

    # quick valid seed (deterministic fallback first so we always have one)
    fb = fallback()
    best_set = fb
    best_max = max(fb)

    # randomized constructions + local search
    for _ in range(400):
        if time.time() - start_time > deadline:
            break
        S = greedy_construct()
        if S is None:
            continue
        S = local_search(S)
        m = max(S)
        if m < best_max:
            best_max = m
            best_set = S[:]
        # early stop if we hit a very small diameter
        if best_max <= 36:
            break

    return best_set
