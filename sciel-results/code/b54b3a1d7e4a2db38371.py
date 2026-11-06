import time
import random
import sys
from math import exp

# EVOLVE-BLOCK-START

# Bit mask of required differences 1..30
REQ_MASK = ((1 << 30) - 1)  # bits 1..30 set

def _diff_mask_for_new(a, x, used_mask):
    """Return (ok, new_mask) for adding x to set a (sorted, contains 0).
    new_mask has bits set for the differences x - v for v in a.
    ok is False if any difference conflicts with used_mask or duplicates within new."""
    new_mask = 0
    for v in a:
        d = x - v
        if d < 1:
            # x must be > all elements; if not, conflict (shouldn't happen for x > a[-1])
            return False, 0
        bit = 1 << d
        if new_mask & bit:
            return False, 0
        if used_mask & bit:
            return False, 0
        new_mask |= bit
    return True, new_mask

def _is_valid(a):
    """Check Sidon property and coverage 1..30."""
    a = sorted(a)
    n = len(a)
    used = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            bit = 1 << d
            if used & bit:
                return False
            used |= bit
    return (used & REQ_MASK) == REQ_MASK

def _cost(a):
    """Cost: conflicts * 1000 + missing * 100 + diameter."""
    a = sorted(a)
    n = len(a)
    used = 0
    conflict = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            bit = 1 << d
            if used & bit:
                conflict += 1
            used |= bit
    missing = bin(REQ_MASK & ~used).count('1')
    diam = a[-1]
    return 1000 * conflict + 100 * missing + diam

def _greedy_build():
    """Coverage-ratio greedy: pick x maximizing newly-covered required diffs."""
    a = [0]
    used_mask = 0
    for _ in range(50):
        covered_mask = used_mask & REQ_MASK
        missing = REQ_MASK & ~covered_mask
        if missing == 0:
            break
        best_x = None
        best_covers = 0
        # Candidate range: up to a reasonable bound
        upper = max(150, a[-1] + 50)
        for x in range(1, upper + 1):
            if x in a:
                continue
            ok, new_mask = _diff_mask_for_new(a, x, used_mask)
            if not ok:
                continue
            new_covers = new_mask & missing
            covers = bin(new_covers).count('1')
            if covers > best_covers:
                best_covers = covers
                best_x = x
            elif covers == best_covers and covers > 0 and best_x is not None and x < best_x:
                best_x = x
        if best_x is None:
            # Fallback: any valid x
            for x in range(1, 500):
                if x in a:
                    continue
                ok, new_mask = _diff_mask_for_new(a, x, used_mask)
                if ok:
                    best_x = x
                    break
            if best_x is None:
                break
        a.append(best_x)
        a.sort()
        # Recompute used_mask
        used_mask = 0
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                used_mask |= (1 << (a[j] - a[i]))
    return a

def _backtrack_mrv(deadline, max_diam=80):
    """MRV backtracking with bitset, ordering by missing-diff coverage."""
    best = None
    best_diam = max_diam

    def covers_all(mask):
        return (mask & REQ_MASK) == REQ_MASK

    def rec(a, used_mask):
        nonlocal best, best_diam
        if time.time() > deadline:
            return
        if covers_all(used_mask):
            if a[-1] < best_diam:
                best_diam = a[-1]
                best = list(a)
            return
        if a[-1] >= best_diam:
            return
        if len(a) > 32:
            return
        missing = REQ_MASK & ~used_mask
        start = a[-1] + 1
        upper = min(best_diam - 1, max_diam)
        # Order candidates by how many missing diffs they cover (MRV-ish)
        candidates = []
        for x in range(start, upper + 1):
            ok, new_mask = _diff_mask_for_new(a, x, used_mask)
            if not ok:
                continue
            new_covers = new_mask & missing
            covers = bin(new_covers).count('1')
            candidates.append((covers, x))
        # Sort: most covering first, then smallest x
        candidates.sort(key=lambda t: (-t[0], t[1]))
        for covers, x in candidates:
            if time.time() > deadline:
                return
            if a[-1] >= best_diam:
                return
            ok, new_mask = _diff_mask_for_new(a, x, used_mask)
            if not ok:
                continue
            a.append(x)
            rec(a, used_mask | new_mask)
            a.pop()

    rec([0], 0)
    return best, best_diam

def _sa_search(initial, deadline, seed=42):
    """Simulated annealing with bitset cost."""
    rng = random.Random(seed)
    current = list(initial)
    current_cost = _cost(current)
    best = list(current)
    best_cost = current_cost
    best_valid = _is_valid(current)

    T = 50.0
    cooling = 0.9995

    while time.time() < deadline:
        n = len(current)
        num_changes = 1 if rng.random() < 0.7 else 2
        trial = list(current)
        for _ in range(num_changes):
            if n < 2:
                break
            idx = rng.randint(1, n - 1)
            center = trial[idx]
            spread = max(10, center // 2)
            new_val = rng.randint(1, center + spread)
            if new_val != trial[idx] and new_val not in trial:
                trial[idx] = new_val
                trial.sort()
        trial_cost = _cost(trial)
        delta = trial_cost - current_cost
        if delta < 0 or rng.random() < exp(-delta / max(T, 0.001)):
            current = trial
            current_cost = trial_cost
            current_valid = _is_valid(current)
            if current_valid and current_cost < best_cost:
                best = list(current)
                best_cost = current_cost
                best_valid = True
        T = max(0.1, T * cooling)
        if rng.random() < 0.01:
            current = list(best)
            current_cost = best_cost
    return best, best_cost, best_valid

def _local_optimize(a, deadline):
    """Reduce each element and try removing elements."""
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    best_valid = _is_valid(a)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each non-zero element
        for i in range(len(a) - 1, 0, -1):
            if time.time() > deadline:
                break
            trial = a[:i] + a[i + 1:]
            if _is_valid(trial):
                a = trial
                best = list(a)
                best_diam = a[-1]
                best_valid = True
                improved = True
                break
        # Try reducing each element
        for i in range(len(a) - 1, 0, -1):
            if time.time() > deadline:
                break
            old = a[i]
            for new_val in range(1, old):
                if new_val in a:
                    continue
                trial = list(a)
                trial[i] = new_val
                trial.sort()
                if _is_valid(trial):
                    if trial[-1] < best_diam:
                        a = trial
                        best = list(a)
                        best_diam = trial[-1]
                        best_valid = True
                        improved = True
                        break
    return best, best_diam, best_valid

def _random_restart_search(deadline):
    """Multiple random restarts with SA."""
    best_set = None
    best_diam = float('inf')
    best_valid = False

    rng = random.Random(777)
    a0 = _greedy_build()
    if _is_valid(a0):
        best_set = a0
        best_diam = a0[-1]
        best_valid = True

    seeds = [1, 2, 3, 42, 123, 456, 789, 1000, 2000, 3000, 5000, 7777, 9999, 12345, 54321]
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        a = [0]
        used_mask = 0
        n_target = rng.randint(7, 12)
        for _ in range(n_target - 1):
            attempts = 0
            while attempts < 100:
                x = rng.randint(1, 120)
                if x in a:
                    attempts += 1
                    continue
                ok, new_mask = _diff_mask_for_new(a, x, used_mask)
                if ok:
                    a.append(x)
                    a.sort()
                    used_mask |= new_mask
                    break
                attempts += 1
        rng_sa = random.Random(seed)
        T = 30.0
        cooling = 0.9995
        current = list(a)
        current_cost = _cost(current)
        max_iters = 200000
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            n = len(current)
            if n < 2:
                break
            idx = rng_sa.randint(1, n - 1)
            center = current[idx]
            spread = max(5, center // 3)
            new_val = rng_sa.randint(1, center + spread)
            if new_val == current[idx] or new_val in current:
                continue
            trial = list(current)
            trial[idx] = new_val
            trial.sort()
            trial_cost = _cost(trial)
            delta = trial_cost - current_cost
            if delta < 0 or rng_sa.random() < exp(-delta / max(T, 0.01)):
                current = trial
                current_cost = trial_cost
                if _is_valid(current) and current[-1] < best_diam:
                    best_set = list(current)
                    best_diam = current[-1]
                    best_valid = True
                    if best_diam <= 50:
                        break
            T = max(0.5, T * cooling)
        if best_set is not None and time.time() < deadline - 2:
            opt, opt_diam, opt_valid = _local_optimize(best_set, deadline - 2)
            if opt_valid and opt_diam < best_diam:
                best_set = opt
                best_diam = opt_diam
    return best_set, best_diam, best_valid

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (evaluator kills at 150s)

    best_set = None
    best_diam = float('inf')

    # Phase 1: MRV backtracking for small-diameter solutions
    if time.time() < deadline - 60:
        bt_result, bt_diam = _backtrack_mrv(deadline - 60)
        if bt_result is not None:
            best_set = bt_result
            best_diam = bt_diam

    # Phase 2: Random restart SA search
    if time.time() < deadline - 10:
        sa_result, sa_diam, sa_valid = _random_restart_search(deadline - 10)
        if sa_result is not None and sa_valid:
            if best_set is None or sa_diam < best_diam:
                best_set = sa_result
                best_diam = sa_diam

    # Phase 3: Local optimization (remove + reduce elements)
    if best_set is not None and time.time() < deadline - 2:
        opt, opt_diam, opt_valid = _local_optimize(best_set, deadline - 2)
        if opt_valid and opt_diam < best_diam:
            best_set = opt
            best_diam = opt_diam

    # Fallback
    if best_set is None:
        best_set = _greedy_build()
        best_diam = max(best_set)

    return best_set

# EVOLVE-BLOCK-END
