"""Fresh design: greedy set-cover init + local search (redundancy removal,
critical-vertex swaps, simulated-annealing-style random moves) for the
ternary covering code K_3(8,1)."""
import time
import random
import numpy as np

# ----- time budget -----
_DEADLINE = time.time() + 110.0  # safety margin under the 150 s kill


def _time_left():
    return _DEADLINE - time.time()


def _ball_indices(center, q, n):
    """All words within Hamming distance 1 of center (center + q*(n-1) words)."""
    out = [center]
    for pos in range(n):
        base = center.copy()
        for v in range(q):
            if v == center[pos]:
                continue
            base[pos] = v
            out.append(base.copy())
    return out


def construct(q, n, radius):
    global _DEADLINE
    _DEADLINE = time.time() + 110.0
    rng = random.Random(123456789)

    total = q ** n  # 6561 for the task
    # Precompute the ball (as index set) for every possible center.
    # index = sum(coord[i]*q**(n-1-i))
    powq = [1] * (n + 1)
    for i in range(n):
        powq[n - 1 - i] = powq[n - i] * q
    # powq[k] = q**(n-1-k)

    def idx_of(word):
        s = 0
        for i in range(n):
            s += word[i] * powq[n - 1 - i]
        return s

    def word_of(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx // powq[n - 1 - i]
            idx %= powq[n - 1 - i]
        return w

    # ball_indices: list of lists of indices
    ball = [None] * total
    for c in range(total):
        w = word_of(c)
        inds = [c]
        for pos in range(n):
            for v in range(q):
                if v == w[pos]:
                    continue
                w[pos] = v
                inds.append(idx_of(w))
            w[pos] = w[pos]  # restore (already)
        ball[c] = inds

    # Greedy set cover: pick center covering most uncovered points.
    uncovered = bytearray([1]) * total
    chosen = []
    while True:
        if _time_left() < 3.0:
            break
        # find center maximizing newly covered count
        best_c = -1
        best_gain = -1
        for c in range(total):
            g = 0
            for p in ball[c]:
                if uncovered[p]:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_c = c
            if g == total:
                break
        if best_gain <= 0:
            break
        chosen.append(best_c)
        for p in ball[best_c]:
            uncovered[p] = 0

    if not chosen:
        return [word_of(0)]

    # coverage count per point
    cover = np.zeros(total, dtype=np.int32)
    for c in chosen:
        for p in ball[c]:
            cover[p] += 1

    def is_covering():
        return np.all(cover >= 1)

    # Local search
    best = list(chosen)
    best_size = len(chosen)

    # helper: try removing a redundant center
    def try_remove(idx):
        nonlocal cover
        c = chosen[idx]
        ok = True
        for p in ball[c]:
            if cover[p] == 1:
                ok = False
                break
        if not ok:
            return False
        for p in ball[c]:
            cover[p] -= 1
        chosen.pop(idx)
        return True

    # pass 1: remove all redundancies
    changed = True
    while changed and _time_left() > 2.0:
        changed = False
        i = 0
        while i < len(chosen):
            if try_remove(i):
                changed = True
            else:
                i += 1

    # pass 2: swaps and random moves (annealing-like)
    def add_center(c):
        for p in ball[c]:
            cover[p] += 1
        chosen.append(c)

    def remove_center(idx):
        c = chosen[idx]
        for p in ball[c]:
            cover[p] -= 1
        chosen.pop(idx)

    # critical vertices: covered exactly once
    # Build, for each center, the set of critical points it uniquely covers.
    # A center is "removable" only if it covers no critical point.
    # A swap: remove center a (must be removable) then add center b that covers
    # at least one currently-uncovered-after-removal point.

    # We'll iterate: repeatedly find a removable center and a good replacement.
    while _time_left() > 4.0:
        # recompute critical mask
        crit = (cover == 1)
        # find removable centers
        removable = []
        for i, c in enumerate(chosen):
            rem = True
            for p in ball[c]:
                if crit[p]:
                    rem = False
                    break
            if rem:
                removable.append(i)
        if not removable:
            break
        # pick a random removable center to try removing
        i = rng.randrange(len(removable))
        c = chosen[i]
        # tentatively remove
        for p in ball[c]:
            cover[p] -= 1
        chosen.pop(i)
        # now some points may be uncovered; we must re-cover them.
        uncovered_pts = [p for p in ball[c] if cover[p] == 0]
        if uncovered_pts:
            # find a center that covers the most uncovered_pts
            best_c = -1
            best_gain = -1
            for cand in range(total):
                g = 0
                for p in ball[cand]:
                    if cover[p] == 0:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_c = cand
            # accept if it covers all uncovered (size stays same) or
            # reduces total size (impossible since we removed one and add one)
            # Only accept if it fully re-covers (keeps size), or we revert.
            if best_gain >= len(uncovered_pts):
                add_center(best_c)
                # after re-adding, try to remove again (shrink)
                # (loop continues)
            else:
                # revert
                chosen.insert(i, c)
                for p in ball[c]:
                    cover[p] += 1
        else:
            # removing it kept coverage -> shrink!
            pass

        if len(chosen) < best_size:
            best_size = len(chosen)
            best = list(chosen)

    # final cleanup: remove redundancies
    changed = True
    while changed and _time_left() > 1.0:
        changed = False
        i = 0
        while i < len(chosen):
            if try_remove(i):
                changed = True
            else:
                i += 1

    if len(chosen) < best_size:
        best_size = len(chosen)
        best = list(chosen)

    # Return words for the best solution found.
    # Use 'best' if it is valid; otherwise fall back to 'chosen' if valid;
    # else to the greedy 'chosen' at the end.
    result = best
    # verify best is covering using cover counts? cover corresponds to 'chosen',
    # not 'best'. Rebuild coverage for 'best' to validate.
    cover_b = np.zeros(total, dtype=np.int32)
    for c in result:
        for p in ball[c]:
            cover_b[p] += 1
    if not np.all(cover_b >= 1):
        # fall back to 'chosen'
        result = chosen

    return [word_of(c) for c in result]
