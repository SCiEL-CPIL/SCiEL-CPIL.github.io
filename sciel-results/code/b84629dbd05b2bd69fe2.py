"""Small inclusion-maximal Sidon sets in [1,256] via weighted set-cover + local shrink."""
import time
import random

N = 256
_DEADLINE = 55.0


def _is_sidon(S):
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _is_maximal(S, N):
    if not S:
        return False
    if not _is_sidon(S):
        return False
    Sset = set(S)
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            sums.add(S[i] + S[j])
    for x in range(1, N + 1):
        if x in Sset:
            continue
        # x is addable iff no new sum collides
        if 2 * x in sums:
            continue
        addable = True
        for s in S:
            if x + s in sums:
                addable = False
                break
        if addable:
            return False
    return True


def _greedy_cover_construct(N, seed, deadline):
    """
    Weighted set-cover greedy.

    A chosen element x covers (hitting) the hyperedges H_s = {y : 2y == s or exists s' in S: y+s' == s}
    for every sum s in its new-sums. To guarantee maximality we need every missing element y
    to be covered by at least one hyperedge from a sum that is "present". We model:
      - universe = missing elements that are currently addable (would violate if we don't block them)
      - each candidate x covers the set of addable y that become non-addable once x is added.
    We pick x maximizing |covered|, tie-broken by smaller x and randomness.
    """
    rng = random.Random(seed)
    S = []
    Sset = set()
    sums = set()

    # candidates = elements not yet chosen
    remaining = list(range(1, N + 1))

    while remaining and time.time() < deadline:
        # Determine which remaining elements are currently addable
        # and, for each addable candidate x, how many addable elements it would block.
        # To be efficient, compute addable set first.
        addable = []
        for x in remaining:
            if 2 * x in sums:
                continue
            ok = True
            for s in S:
                if x + s in sums:
                    ok = False
                    break
            if ok:
                addable.append(x)

        if not addable:
            break

        best_x = None
        best_cov = -1
        best_cov_set = None
        # sample for speed when many candidates
        if len(addable) > 40:
            cand = rng.sample(addable, 40)
        else:
            cand = addable

        for x in cand:
            new_sums = {2 * x} | {x + s for s in S}
            # blocked = addable y (y != x) that become non-addable after adding x
            blocked = 0
            blocked_set = set()
            for y in addable:
                if y == x:
                    continue
                # y non-addable after adding x iff 2y in sums|new_sums or exists s in S|{x}: y+s in sums|new_sums
                if 2 * y in sums or 2 * y in new_sums:
                    blocked += 1
                    blocked_set.add(y)
                    continue
                hit = False
                for s in S:
                    if y + s in sums or y + s in new_sums:
                        hit = True
                        break
                if not hit and (y + x) in sums or (y + x) in new_sums:
                    hit = True
                if hit:
                    blocked += 1
                    blocked_set.add(y)
            if blocked > best_cov:
                best_cov = blocked
                best_x = x
                best_cov_set = blocked_set

        if best_x is None:
            # no improvement found; just add a random addable to make progress
            best_x = rng.choice(addable)
            best_cov_set = set()

        S.append(best_x)
        Sset.add(best_x)
        sums.update({2 * best_x} | {best_x + s for s in S[:-1]})
        # remove chosen and blocked from remaining
        if best_cov_set:
            remaining = [z for z in remaining if z != best_x and z not in best_cov_set]
        else:
            remaining = [z for z in remaining if z != best_x]

    S.sort()
    return S


def _shrink(S, N, deadline):
    """Try to remove elements while keeping maximality."""
    S = list(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(S)):
            cand = S[:i] + S[i + 1:]
            if not cand:
                continue
            if _is_maximal(cand, N):
                S = cand
                improved = True
                break
    return S


def construct():
    start = time.time()
    best = None
    best_size = N + 1

    def elapsed():
        return time.time() - start

    # Phase 1: multiple greedy set-cover restarts
    for seed in range(400):
        if elapsed() > _DEADLINE - 20:
            break
        S = _greedy_cover_construct(N, seed, deadline=_DEADLINE - 15)
        if not S:
            continue
        if len(S) < best_size:
            if elapsed() < _DEADLINE - 10:
                S2 = _shrink(S, N, deadline=_DEADLINE - 10)
                if len(S2) < len(S):
                    S = S2
            if len(S) < best_size:
                best = S
                best_size = len(S)
                # early stop if we hit a very small size
                if best_size <= 12:
                    break

    # Phase 2: aggressive shrink on the best
    if best is not None and elapsed() < _DEADLINE - 3:
        for _ in range(8):
            if elapsed() > _DEADLINE - 3:
                break
            S2 = _shrink(best, N, deadline=_DEADLINE - 3)
            if len(S2) < len(best):
                best = S2
                best_size = len(S2)

    # Fallback: plain randomized greedy to always return something valid-ish
    if best is None:
        rng = random.Random(0)
        cands = list(range(1, N + 1))
        rng.shuffle(cands)
        S = []
        sums = set()
        for x in cands:
            ns = [x + a for a in S] + [2 * x]
            if len(set(ns)) == len(ns) and not sums.intersection(set(ns)):
                S.append(x)
                sums.update(ns)
        S.sort()
        best = S

    return best
