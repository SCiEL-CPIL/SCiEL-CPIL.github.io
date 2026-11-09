"""Find maximum cardinality dissociated reciprocal subset of [1,22].

Combines the multi-phase search (greedy, randomized, simulated annealing,
local search) with a fast modular filter: before the exact integer subset-sum
distinctness check, we verify distinctness modulo a large prime P.  Distinct
subset sums mod P is a NECESSARY condition for distinct integer sums, so a
collision mod P immediately rejects a candidate without the expensive exact
check.  This dramatically increases the number of candidates the search can
evaluate within the wall-clock budget.
"""
import time
import random
import math

# EVOLVE-BLOCK-START

# ---------- Constants ----------
_LCM = 232792560                       # lcm(1..22)
_P = 10**9 + 7                         # large prime for modular filter
# Precompute integer weights w_i = LCM/i and modular residues r_i = (LCM/i) mod P
_W = [_LCM // i for i in range(1, 23)]
_R = [w % _P for w in _W]

# ---------- Modular fast filter ----------
def _mod_dissociated(indices, _R=_R, _P=_P):
    """Fast necessary-condition check: all 2^n subset sums of the modular
    residues are distinct mod P.  Returns True only if no collision mod P.
    This is a NECESSARY (not sufficient) condition for exact dissociation."""
    n = len(indices)
    if n == 0:
        return True
    sums = [0]
    for idx in indices:
        r = _R[idx - 1]
        new_sums = [(s + r) % _P for s in sums]
        all_sums = sums + new_sums
        if len(set(all_sums)) != len(all_sums):
            return False
        sums = all_sums
    return True

# ---------- Exact integer check ----------
def _exact_dissociated(indices, _W=_W):
    """Exact check: all 2^n subset sums of integer weights are distinct."""
    n = len(indices)
    if n == 0:
        return True
    sums = [0]
    for idx in indices:
        w = _W[idx - 1]
        new_sums = [s + w for s in sums]
        all_sums = sums + new_sums
        if len(set(all_sums)) != len(all_sums):
            return False
        sums = all_sums
    return True

def _is_dissociated(indices):
    """Two-stage check: fast modular filter, then exact integer check."""
    if not _mod_dissociated(indices):
        return False
    return _exact_dissociated(indices)

# ---------- Min gap (for tie-breaking, only for small sets) ----------
def _min_gap(indices, _W=_W):
    """Minimum positive difference between any two distinct subset sums.
    Returns 0 on collision.  Only used for sets up to size 14."""
    n = len(indices)
    if n == 0:
        return float('inf')
    if n > 14:
        return 1  # sentinel; too expensive to compute exactly
    sums = [0]
    for idx in indices:
        w = _W[idx - 1]
        new_sums = [s + w for s in sums]
        all_sums = sums + new_sums
        if len(set(all_sums)) != len(all_sums):
            return 0
        sums = all_sums
    sums_sorted = sorted(sums)
    min_gap = float('inf')
    for i in range(1, len(sums_sorted)):
        gap = sums_sorted[i] - sums_sorted[i - 1]
        if gap < min_gap:
            min_gap = gap
    return min_gap

# ---------- Greedy construction ----------
def _greedy_construct(pool, deadline):
    """Greedy: add elements one by one if they maintain dissociation.
    Uses the modular filter as a fast pre-check."""
    current = []
    for elem in pool:
        if time.time() > deadline:
            break
        candidate = current + [elem]
        # Fast modular pre-check
        if not _mod_dissociated(candidate):
            continue
        # Exact check
        if _exact_dissociated(candidate):
            current.append(elem)
    return current

def _randomized_greedy(pool, deadline, n_restarts=500):
    """Multiple randomized greedy attempts with modular filtering."""
    best = []
    for _ in range(n_restarts):
        if time.time() > deadline:
            break
        shuffled = list(pool)
        random.shuffle(shuffled)
        result = _greedy_construct(shuffled, deadline)
        if len(result) > len(best):
            best = result
    return best

# ---------- Simulated annealing ----------
def _simulated_annealing(initial_set, all_elements, deadline, n_iterations=200000):
    """Simulated annealing to find larger dissociated sets.
    Uses modular filter to reject most bad moves instantly."""
    current = list(initial_set)
    current_gap = _min_gap(current) if current else float('inf')
    best = list(current)
    best_gap = current_gap
    best_size = len(current)

    T = 1.0
    cooling = 0.99995

    for it in range(n_iterations):
        if time.time() > deadline:
            break
        if T < 1e-12:
            break

        move_type = random.random()
        new_set = list(current)

        if move_type < 0.5:
            # Swap: remove one from current, add one from outside
            if len(current) == 0:
                if all_elements:
                    new_set.append(random.choice(all_elements))
                else:
                    continue
            elif not all_elements:
                if current:
                    idx = random.randrange(len(current))
                    new_set.pop(idx)
                else:
                    continue
            else:
                remove_idx = random.randrange(len(current))
                outside = [e for e in all_elements if e not in current]
                if not outside:
                    new_set.pop(remove_idx)
                else:
                    new_set.pop(remove_idx)
                    new_set.append(random.choice(outside))
        elif move_type < 0.75:
            # Add an element
            outside = [e for e in all_elements if e not in current]
            if outside:
                new_set.append(random.choice(outside))
            else:
                continue
        else:
            # Remove an element
            if current:
                idx = random.randrange(len(current))
                new_set.pop(idx)
            else:
                continue

        # Fast modular pre-check
        mod_ok = _mod_dissociated(new_set)
        if not mod_ok:
            # Collision mod P: almost certainly a real collision
            if random.random() > 0.01 * T:
                T *= cooling
                continue
            # Very rarely accept to escape local minima
            new_dissociated = _exact_dissociated(new_set)
            if not new_dissociated:
                T *= cooling
                continue
        else:
            new_dissociated = _exact_dissociated(new_set)

        new_size = len(new_set)
        new_gap = _min_gap(new_set) if new_dissociated else 0

        # Objective: primarily cardinality, secondarily gap
        if new_dissociated:
            old_score = len(current) * 10000 + (current_gap if current_gap != float('inf') else 0)
            new_score = new_size * 10000 + new_gap
        else:
            old_score = len(current) * 10000 + (current_gap if current_gap != float('inf') else 0)
            new_score = -1000000

        delta = new_score - old_score

        if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
            current = new_set
            current_gap = new_gap
            if new_dissociated:
                if new_size > best_size or (new_size == best_size and new_gap > best_gap):
                    best = list(current)
                    best_gap = new_gap
                    best_size = new_size

        T *= cooling

    return best

# ---------- Local search refinement ----------
def _local_search_refine(initial_set, all_elements, deadline, max_iter=50000):
    """Local search: try all single additions and swaps to improve.
    Uses modular filter to skip obviously bad candidates."""
    current = list(initial_set)
    improved = True
    iter_count = 0
    while improved and time.time() < deadline and iter_count < max_iter:
        improved = False
        iter_count += 1
        current_dissociated = _exact_dissociated(current)
        if not current_dissociated:
            break

        current_gap = _min_gap(current) if current else float('inf')

        # Try adding each element
        for elem in all_elements:
            if elem in current:
                continue
            new_set = current + [elem]
            # Fast modular pre-check
            if not _mod_dissociated(new_set):
                continue
            if _exact_dissociated(new_set):
                new_gap = _min_gap(new_set)
                if len(new_set) > len(current) or (len(new_set) == len(current) and new_gap > current_gap):
                    current = new_set
                    current_gap = new_gap
                    improved = True
                    break

        # Try swaps
        if not improved:
            for i, elem_out in enumerate(current):
                for elem_in in all_elements:
                    if elem_in in current:
                        continue
                    new_set = current[:i] + current[i + 1:] + [elem_in]
                    if not _mod_dissociated(new_set):
                        continue
                    if _exact_dissociated(new_set):
                        new_gap = _min_gap(new_set)
                        if len(new_set) > len(current) or (len(new_set) == len(current) and new_gap > current_gap):
                            current = new_set
                            current_gap = new_gap
                            improved = True
                            break
                if improved:
                    break

    return current

# ---------- Main construct ----------
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit

    all_elements = list(range(1, 23))

    # Known valid 9-element set
    seed_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = list(seed_set)

    # Phase 1: Greedy from seed with different orderings
    remaining = [i for i in range(1, 23) if i not in seed_set]

    for order in [remaining, sorted(remaining, reverse=True), sorted(remaining)]:
        if time.time() > deadline:
            break
        current = list(seed_set)
        for elem in order:
            if time.time() > deadline:
                break
            candidate = current + [elem]
            if _mod_dissociated(candidate) and _exact_dissociated(candidate):
                current.append(elem)
        if len(current) > len(best):
            best = list(current)

    # Phase 2: Randomized greedy with many restarts (modular filter makes this fast)
    if time.time() < deadline:
        rand_best = _randomized_greedy(all_elements, deadline, n_restarts=2000)
        if len(rand_best) > len(best):
            best = rand_best

    # Phase 3: Simulated annealing to explore larger sets
    if time.time() < deadline:
        sa_deadline = min(deadline, time.time() + 35)
        sa_best = _simulated_annealing(best, all_elements, sa_deadline, n_iterations=300000)
        if len(sa_best) > len(best):
            best = sa_best

    # Phase 4: Local search refinement
    if time.time() < deadline:
        ls_best = _local_search_refine(best, all_elements, deadline, max_iter=80000)
        if len(ls_best) > len(best):
            best = ls_best

    # Phase 5: More randomized greedy if time remains
    if time.time() < deadline:
        more_best = _randomized_greedy(all_elements, deadline, n_restarts=1000)
        if len(more_best) > len(best):
            best = more_best

    # Phase 6: Final local search on best
    if time.time() < deadline and len(best) < 16:
        final_best = _local_search_refine(best, all_elements, deadline, max_iter=80000)
        if len(final_best) > len(best):
            best = final_best

    # Phase 7: One more SA round if still time
    if time.time() < deadline and len(best) < 16:
        sa2_deadline = min(deadline, time.time() + 10)
        sa2_best = _simulated_annealing(best, all_elements, sa2_deadline, n_iterations=100000)
        if len(sa2_best) > len(best):
            best = sa2_best

    return best

# EVOLVE-BLOCK-END
