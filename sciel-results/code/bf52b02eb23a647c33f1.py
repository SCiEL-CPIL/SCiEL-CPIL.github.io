import time
from fractions import Fraction
from itertools import combinations

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


def _try_ilp_search():
    """Use scipy.optimize.milp to search for a valid set of size 20."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
    except ImportError:
        return []

    # Precompute all reciprocal identities: for each a in [1,24], find all subsets
    # T of [1,24]\{a} such that 1/a = sum_{t in T} 1/t.
    # For each such (a, T), we add a constraint: x_a + sum_{t in T} x_t <= |T|
    # This prevents the case where a is selected AND all elements of T are selected.
    
    # But wait, the constraint should be: NOT (x_a = 1 AND x_t = 1 for all t in T)
    # Which is equivalent to: x_a + sum_{t in T} (1 - x_t) >= 1
    # Or: sum_{t in T} x_t - x_a <= |T| - 1
    
    # Actually, let's think again. We want to forbid the configuration where
    # a is selected and all elements of T are selected.
    # This is a clause: NOT (x_a AND x_{t1} AND ... AND x_{tk})
    # In ILP, this is: x_a + x_{t1} + ... + x_{tk} <= k (where k = |T|)
    # Because if all are 1, the sum is k+1 > k, which is forbidden.
    
    # However, generating all such subsets T is expensive. Let's be smarter.
    # For a given a, 1/a = sum_{t in T} 1/t. Since 1/t >= 1/24, we have |T| <= 24/a.
    # For a=1, |T| <= 24. For a=2, |T| <= 12. Etc.
    # So the subsets T are not too large.
    
    # Let's precompute the conflicts.
    conflicts = []  # List of (a, T) where T is a frozenset of elements
    
    for a in range(1, 25):
        target = Fraction(1, a)
        # Find all subsets T of [1,24]\{a} that sum to target.
        # Use meet-in-the-middle.
        others = [x for x in range(1, 25) if x != a]
        n = len(others)
        if n == 0:
            continue
        mid = n // 2
        left, right = others[:mid], others[mid:]
        
        # Right half subset sums
        R = {Fraction(0): frozenset()}
        for v in right:
            new_R = dict(R)
            for s, T in R.items():
                ns = s + Fraction(1, v)
                new_R[ns] = T | {v}
            R = new_R
        
        # Left half subset sums, check if target - l is in R
        L = {Fraction(0): frozenset()}
        for v in left:
            new_L = dict(L)
            for s, T in L.items():
                ns = s + Fraction(1, v)
                new_L[ns] = T | {v}
                # Check if ns is a conflict
                rem = target - ns
                if rem in R:
                    conflicts.append((a, T | R[rem]))
            L = new_L
        
        # Also check if target itself is in R (empty left subset)
        if target in R:
            conflicts.append((a, R[target]))
    
    # Now build the ILP.
    # Variables: x_1, ..., x_24 (binary)
    # Objective: maximize sum x_i
    # Constraints: for each (a, T) in conflicts: x_a + sum_{t in T} x_t <= |T|
    
    # Remove duplicate conflicts
    conflicts = list(set(conflicts))
    
    if not conflicts:
        # No conflicts found, so any set is valid? That can't be right.
        # Actually, if no conflicts, then the maximum set is [1,24].
        return list(range(1, 25))
    
    # Build constraint matrix
    m = len(conflicts)
    c = np.zeros((m, 24))
    lb = np.zeros(m)
    ub = np.zeros(m)
    
    for i, (a, T) in enumerate(conflicts):
        c[i, a - 1] = 1
        for t in T:
            c[i, t - 1] = 1
        lb[i] = 0
        ub[i] = len(T)
    
    # Objective: maximize sum x_i, so c_obj = -1 (minimize -sum)
    c_obj = np.ones(24)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=np.zeros(24), ub=np.ones(24))
    
    # Integer: all variables are binary
    integrality = np.ones(24)
    
    constraints = LinearConstraint(c, lb, ub)
    
    res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0:  # Success
        x = res.x
        # Extract the set
        S = [i + 1 for i in range(24) if x[i] > 0.5]
        if S and is_valid(S):
            return S
    return []


def construct():
    global _START
    _START = time.time()

    # 1) Direct answer: the known size-19 set. Validate exactly.
    ans = list(_BASE)
    if not is_valid(ans):
        # Should not happen; fall back to the size-18 superincreasing set.
        ans = list(range(7, 25))

    # 2) Try ILP to find a size-20 set.
    if not _deadline_hit():
        try:
            ilp_result = _try_ilp_search()
            if ilp_result and len(ilp_result) > len(ans):
                ans = ilp_result
        except Exception:
            pass

    # 3) Use spare time to try to beat 19 (or 20) via local search.
    if not _deadline_hit():
        try:
            improved = _local_search(ans, _DEADLINE)
            if improved and len(improved) > len(ans) and is_valid(improved):
                ans = improved
        except Exception:
            pass

    # 4) Final safety: ensure nonempty and valid.
    if not ans:
        ans = list(range(7, 25))
    if not is_valid(ans):
        ans = list(range(7, 25))
    return list(ans)


if __name__ == "__main__":
    print(construct())
    print("size:", len(construct()))
