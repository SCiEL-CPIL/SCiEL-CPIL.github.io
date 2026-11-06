import time
from fractions import Fraction

# Known valid size-19 set: [5,6,8,9,...,24]
_BASE = [5, 6] + list(range(8, 25))

# Safety margin: evaluator kills at 90 s; stop at 85 s.
_DEADLINE = 85.0
_START = None


def _elapsed():
    return 0.0 if _START is None else (time.time() - _START)


def _deadline_hit():
    return _elapsed() > _DEADLINE


def _subset_sum_possible(vals, target):
    """Exact: some subset of `vals` (list of Fraction) sums to `target`?
    Meet-in-the-middle O(2^(n/2)) with memoized right-half sums."""
    n = len(vals)
    if n == 0:
        return target == 0
    if n == 1:
        return vals[0] == target
    mid = n // 2
    left, right = vals[:mid], vals[mid:]

    # right-half subset sums
    R = {Fraction(0)}
    for v in right:
        R = R | {x + v for x in R}

    # left-half subset sums, early-exit if (target - l) in R
    L = {Fraction(0)}
    if target in R:
        return True
    for v in left:
        new = set()
        for x in L:
            y = x + v
            if (target - y) in R:
                return True
            new.add(y)
        L |= new
    return False


def _violations(S):
    """Count of elements a in S for which 1/a IS a subset-sum of the others."""
    S = list(S)
    r = [Fraction(1, v) for v in S]
    cnt = 0
    for i in range(len(S)):
        others = r[:i] + r[i + 1:]
        if _subset_sum_possible(others, r[i]):
            cnt += 1
    return cnt


def is_valid(S):
    S = list(set(S))
    if not S:
        return False
    return _violations(S) == 0


def _local_search(S, deadline):
    """Bounded local search from S: try swaps to reduce violations; accept
    improvements. Returns best valid set found (or best-effort)."""
    cur = list(set(S))
    best = cur if is_valid(cur) else []
    pool = list(range(1, 25))

    def score_of(x):
        # prefer valid sets, then larger, then fewer violations
        v = _violations(x)
        if v == 0:
            return (0, -len(x), 0)
        return (1, v, -len(x))

    cur_score = score_of(cur)
    stuck = 0
    while not _deadline_hit():
        improved = False
        cur_set = set(cur)
        # candidate moves: add an element not in set
        for a in pool:
            if _deadline_hit():
                break
            if a in cur_set:
                continue
            trial = cur + [a]
            ts = score_of(trial)
            if ts < cur_score:
                cur, cur_score = trial, ts
                if ts[0] == 0 and not best:
                    best = list(cur)
                improved = True
        if _deadline_hit():
            break
        # drop one element
        for a in list(cur):
            if _deadline_hit():
                break
            trial = [x for x in cur if x != a]
            ts = score_of(trial)
            if ts < cur_score:
                cur, cur_score = trial, ts
                if ts[0] == 0 and (not best or len(cur) > len(best)):
                    best = list(cur)
                improved = True
        if _deadline_hit():
            break
        # swap: add a, drop b (a not in set, b in set)
        for a in pool:
            if _deadline_hit():
                break
            if a in cur:
                continue
            for b in list(cur):
                if _deadline_hit():
                    break
                trial = [x for x in cur if x != b] + [a]
                ts = score_of(trial)
                if ts < cur_score:
                    cur, cur_score = trial, ts
                    if ts[0] == 0 and (not best or len(cur) > len(best)):
                        best = list(cur)
                    improved = True
        if not improved:
            stuck += 1
            if stuck >= 4:
                break
        else:
            stuck = 0

    if is_valid(cur) and (not best or len(cur) > len(best)):
        best = list(cur)
    return best


def construct():
    global _START
    _START = time.time()

    # 1) Direct answer: the known size-19 set. Validate exactly.
    ans = list(_BASE)
    if not is_valid(ans):
        # Should not happen; fall back to the size-18 superincreasing set.
        ans = list(range(7, 25))

    # 2) Use spare time to try to beat 19 (hunt for size 20) via local search.
    if not _deadline_hit():
        try:
            improved = _local_search(ans, _DEADLINE)
            if improved and len(improved) > len(ans) and is_valid(improved):
                ans = improved
        except Exception:
            pass

    # 3) Final safety: ensure nonempty and valid.
    if not ans:
        ans = list(range(7, 25))
    if not is_valid(ans):
        ans = list(range(7, 25))
    return list(ans)


if __name__ == "__main__":
    print(construct())
    print("size:", len(construct()))
