"""Difference triangle set (7,5): minimize largest entry with all positive
differences distinct across all rows. Fresh design: set-packing / CSP over
'difference frames' with bitmask compatibility, MRV heuristic, randomized
restarts, iterative deepening, and a hard wall-clock deadline guard."""

import time
import random
from itertools import combinations

# Hard safety margin: the evaluator kills at 150s; stop searching well before.
_DEADLINE = 130.0


def _frames_for_scope(k, M):
    """All strictly increasing (k+1)-tuples starting at 0 with max == M,
    sorted by scope. Each returned as a tuple."""
    k1 = k + 1
    # choose the other k entries from 1..M-1
    out = []
    for combo in combinations(range(1, M), k):
        out.append((0,) + combo)
    return out


def _diff_mask(frame, M):
    """Bitmask of positive differences of a frame, over universe 1..M."""
    mask = 0
    n = len(frame)
    for i in range(n):
        ai = frame[i]
        for j in range(i + 1, n):
            d = frame[j] - ai
            mask |= (1 << d)
    return mask


def _build_frames(k, M):
    """Return (frames, masks, compat) for scope M.
    frames: list of tuples; masks: list of int bitmasks;
    compat[i]: int bitset of frames j with masks[i] & masks[j] == 0."""
    frames = _frames_for_scope(k, M)
    masks = [_diff_mask(f, M) for f in frames]
    n = len(frames)
    compat = []
    for i in range(n):
        mi = masks[i]
        bits = 0
        for j in range(n):
            if mi & masks[j]:
                continue
            bits |= (1 << j)
        compat.append(bits)
    return frames, masks, compat


def _solve_scope(k, n_rows, M, frames, masks, compat, rng, time_limit):
    """Backtracking with MRV. Returns list of frames (tuples) or None."""
    n = len(frames)
    full = (1 << n) - 1
    used = 0  # bitmask of used differences
    chosen = [None] * n_rows

    # Precompute for each frame its compat as a set of candidate indices
    # (for MRV counting we just count bits of compat[i] & available).
    # available: bitset of frames still allowed (i.e. disjoint with used)
    def available_mask():
        # frame i is available if masks[i] & used == 0
        av = 0
        for i in range(n):
            if masks[i] & used:
                continue
            av |= (1 << i)
        return av

    def mrv_count(av, i):
        # number of frames j (j != i, j in av) compatible with i
        c = compat[i] & av
        c &= ~(1 << i)
        return c.bit_count()

    def dfs(depth, used, av):
        if time.time() > time_limit:
            return False
        if depth == n_rows:
            return True
        # Build candidate list with MRV heuristic
        cands = []
        av_list = av
        # iterate over set bits in av
        tmp = av_list
        while tmp:
            lsb = tmp & (-tmp)
            i = lsb.bit_length() - 1
            tmp ^= lsb
            # count remaining if we pick i
            c = compat[i] & av_list
            c &= ~(1 << i)
            cnt = c.bit_count()
            cands.append((cnt, i))
        if not cands:
            return False
        # sort by fewest remaining; randomize ties
        cands.sort(key=lambda t: (t[0], rng.random()))
        for cnt, i in cands:
            if cnt < 0:
                continue
            new_used = used | masks[i]
            # new available: frames j in av with masks[j] & new_used == 0 and j != i
            # reuse compat: a frame j is still available after picking i iff
            # (masks[j] & new_used)==0  <=> (masks[j]&used)==0 and (masks[j]&masks[i])==0
            # (masks[j]&used)==0 means j in av; (masks[j]&masks[i])==0 means j in compat[i]
            new_av = av & compat[i]
            new_av &= ~(1 << i)
            chosen[depth] = i
            if dfs(depth + 1, new_used, new_av):
                return True
            chosen[depth] = None
        return False

    av = available_mask()
    if dfs(0, 0, av):
        return [frames[chosen[d]] for d in range(n_rows)]
    return None


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(20250705)

    # Greedy fallback (always valid), used if search can't finish in time.
    def greedy():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best = greedy()
    best_scope = max(max(r) for r in best)

    # Try iterative deepening from the counting bound upward.
    # Counting bound: n_rows * C(k+1,2) differences must fit in 1..M => M(M-1)/2 >= need
    need = n_rows * (k * (k + 1) // 2)
    M_low = 1
    while M_low * (M_low - 1) // 2 < need:
        M_low += 1
    # small practical lower bound for feasibility
    M_low = max(M_low, k + 1)

    # Cap how far we go so we don't spend all time on one giant scope.
    M_high = best_scope  # never need to beat the greedy scope by much

    for M in range(M_low, M_high + 1):
        if time.time() > deadline - 15:
            break
        # Build frames for this scope
        frames, masks, compat = _build_frames(k, M)
        n = len(frames)
        if n < n_rows:
            continue
        # If total compatible capacity is too low, skip
        # (a necessary condition: each frame must have enough partners)
        # Try a few randomized restarts at this scope.
        found = None
        max_restarts = 40
        for r in range(max_restarts):
            if time.time() > deadline - 15:
                break
            res = _solve_scope(k, n_rows, M, frames, masks, compat, rng, deadline - 15)
            if res is not None:
                found = res
                break
        if found is not None:
            # found a valid solution at scope M (which is < best_scope)
            best = [list(f) for f in found]
            best_scope = M
            break  # M is increasing, first success is minimal
    return best


# Self-test (not run by evaluator)
if __name__ == "__main__":
    t0 = time.time()
    res = construct(7, 5)
    print("scope:", max(max(r) for r in res))
    print(res)
    # validate
    used = set()
    ok = True
    for r in res:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in used:
                    ok = False
                used.add(d)
    print("valid:", ok, "time:", time.time() - t0)
