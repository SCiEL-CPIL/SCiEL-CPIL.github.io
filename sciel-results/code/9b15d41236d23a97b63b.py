"""Rewritten program: bitmask conflict model + reduction local search for minimum maximal no-isosceles set."""
import itertools
import time

N = 16
TOTAL = N * N
DEADLINE = 320.0  # safety margin before 330s kill


def _idx(x, y):
    return y * N + x


def _xy(i):
    return (i % N, i // N)


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _build_conflicts():
    """conflict[i][j] = bitmask of points k such that (i,j,k) form an isosceles triangle."""
    pts = [_xy(i) for i in range(TOTAL)]
    conflict = [[0] * TOTAL for _ in range(TOTAL)]
    for i in range(TOTAL):
        pi = pts[i]
        for j in range(i + 1, TOTAL):
            pj = pts[j]
            d = _dist2(pi, pj)
            mask = 0
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                pk = pts[k]
                d1 = _dist2(pi, pk)
                d2 = _dist2(pj, pk)
                # isosceles if any two of (d, d1, d2) are equal
                if d == d1 or d == d2 or d1 == d2:
                    mask |= (1 << k)
            conflict[i][j] = mask
            conflict[j][i] = mask
    return conflict


def _is_maximal(chosen_set, conflict):
    """Every point not in chosen_set must form an isosceles triangle with some pair in chosen_set."""
    for p in range(TOTAL):
        if p in chosen_set:
            continue
        # check if p conflicts with any pair in chosen_set
        bit = 1 << p
        found = False
        for a in chosen_set:
            for b in chosen_set:
                if a >= b:
                    continue
                if conflict[a][b] & bit:
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True


def _is_valid(chosen_set, conflict):
    """No isosceles triangle among chosen points."""
    for a, b in itertools.combinations(chosen_set, 2):
        mask = conflict[a][b]
        # if any chosen point is in mask (other than a,b), invalid
        # mask doesn't include a or b by construction? It excludes k==i or k==j, so a,b not in mask.
        if mask & 0:
            pass
        # check intersection with chosen_set
        inter = False
        for c in chosen_set:
            if c != a and c != b and (mask >> c) & 1:
                inter = True
                break
        if inter:
            return False
    return True


def _greedy_build(conflict):
    """Greedy row-major construction, returns set of indices."""
    chosen = set()
    for p in range(TOTAL):
        bit = 1 << p
        ok = True
        for a, b in itertools.combinations(chosen, 2):
            if conflict[a][b] & bit:
                ok = False
                break
        if ok:
            chosen.add(p)
    return chosen


def _reduce(chosen_set, conflict, deadline):
    """Try to remove points while keeping the set maximal. Return reduced set."""
    chosen = set(chosen_set)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # iterate over points in some order; try removing each
        for p in sorted(chosen):
            if time.time() > deadline:
                break
            if p not in chosen:
                continue
            trial = chosen - {p}
            if len(trial) < 2:
                continue
            if _is_maximal(trial, conflict):
                chosen = trial
                improved = True
                break
    return chosen


def _greedy_reinsert(chosen_set, conflict, deadline):
    """Try to add points that keep validity AND maximality. (Usually not helpful for size, but can enable further reduction.)"""
    chosen = set(chosen_set)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for p in range(TOTAL):
            if time.time() > deadline:
                break
            if p in chosen:
                continue
            # check validity: p doesn't form isosceles with any pair
            bit = 1 << p
            valid = True
            for a, b in itertools.combinations(chosen, 2):
                if conflict[a][b] & bit:
                    valid = False
                    break
            if not valid:
                continue
            # check maximality of new set
            trial = chosen | {p}
            if _is_maximal(trial, conflict):
                chosen = trial
                improved = True
                break
    return chosen


def construct():
    start = time.time()
    deadline = start + DEADLINE

    conflict = _build_conflicts()

    # Initial greedy set
    chosen = _greedy_build(conflict)

    best = len(chosen)
    best_set = set(chosen)

    # Aggressive reduction loop
    while time.time() < deadline:
        # Reduce
        reduced = _reduce(best_set, conflict, deadline)
        if len(reduced) < best:
            best = len(reduced)
            best_set = reduced
        else:
            # No reduction possible; try perturbation: remove a random point and re-greedy? 
            # Instead, do a "kick": pick a point, remove it, then greedily fill to maximal, then reduce.
            if best_set:
                # kick: remove a random point
                import random
                random.seed(12345)
                p = random.choice(sorted(best_set))
                kicked = best_set - {p}
                # greedily add back to make maximal
                # re-greedy from kicked set
                trial = set(kicked)
                for q in range(TOTAL):
                    if time.time() > deadline:
                        break
                    if q in trial:
                        continue
                    bit = 1 << q
                    valid = True
                    for a, b in itertools.combinations(trial, 2):
                        if conflict[a][b] & bit:
                            valid = False
                            break
                    if valid:
                        trial.add(q)
                # now reduce
                reduced2 = _reduce(trial, conflict, deadline)
                if len(reduced2) < best:
                    best = len(reduced2)
                    best_set = reduced2
            break

    # Convert to list of (x, y)
    result = [_xy(i) for i in best_set]
    return result


# Precompute conflicts at import time to spread cost
_conflict_cache = None


def _get_conflict():
    global _conflict_cache
    if _conflict_cache is None:
        _conflict_cache = _build_conflicts()
    return _conflict_cache


# Override construct to use cached conflicts
def construct():
    start = time.time()
    deadline = start + DEADLINE

    conflict = _get_conflict()

    # Initial greedy set
    chosen = _greedy_build(conflict)

    best = len(chosen)
    best_set = set(chosen)

    # Aggressive reduction loop
    import random
    rng = random.Random(42)

    while time.time() < deadline:
        # Reduce
        reduced = _reduce(best_set, conflict, deadline)
        if len(reduced) < best:
            best = len(reduced)
            best_set = reduced
        else:
            # No reduction; do a kick
            if len(best_set) > 2:
                p = rng.choice(sorted(best_set))
                kicked = best_set - {p}
                trial = set(kicked)
                for q in range(TOTAL):
                    if time.time() > deadline:
                        break
                    if q in trial:
                        continue
                    bit = 1 << q
                    valid = True
                    for a, b in itertools.combinations(trial, 2):
                        if conflict[a][b] & bit:
                            valid = False
                            break
                    if valid:
                        trial.add(q)
                reduced2 = _reduce(trial, conflict, deadline)
                if len(reduced2) < best:
                    best = len(reduced2)
                    best_set = reduced2
            else:
                break

    result = [_xy(i) for i in best_set]
    return result
