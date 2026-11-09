"""Combined approach: collision-driven greedy + local search + SA polish on min subset-sum gap."""
import time
import random

_LCM = 232792560

def _subset_sums(scaled):
    """Return list of all 2^k subset sums for the given scaled weights."""
    k = len(scaled)
    n = 1 << k
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        sums[mask] = sums[mask ^ lb] + scaled[idx]
    return sums

def _verify(subset):
    """Check if all 2^k subset sums of {LCM//x for x in subset} are distinct."""
    k = len(subset)
    if k == 0:
        return True
    scaled = [_LCM // x for x in subset]
    sums = _subset_sums(scaled)
    return len(set(sums)) == len(sums)

def _min_gap(subset):
    """Return the minimum absolute difference between distinct subset sums.
    Returns inf if dissociated (no collisions)."""
    k = len(subset)
    if k == 0:
        return float('inf')
    scaled = [_LCM // x for x in subset]
    sums = _subset_sums(scaled)
    unique = sorted(set(sums))
    if len(unique) == len(sums):
        return float('inf')
    # If there are collisions, min gap is 0
    # But for SA we want a continuous measure: use the smallest positive gap among distinct values
    # Actually, if there's a collision, gap is 0. For SA we want to maximize the min gap.
    # Let's return 0 if collision, else the min positive gap.
    # For a smooth surrogate, we can use a soft-min of all pairwise differences.
    # But that's O(n^2) in the number of subset sums. For k<=12, n=4096, that's ~8M pairs - too slow.
    # Instead, use the min gap between sorted unique sums as a proxy.
    gaps = [unique[i+1] - unique[i] for i in range(len(unique)-1)]
    return min(gaps) if gaps else float('inf')

def _find_collision(subset):
    """Find two different subsets with the same sum, or None if dissociated."""
    k = len(subset)
    if k == 0:
        return None
    scaled = [_LCM // x for x in subset]
    sums = _subset_sums(scaled)
    seen = {}
    for mask in range(len(sums)):
        s = sums[mask]
        if s in seen:
            return seen[s], mask
        seen[s] = mask
    return None

def _greedy_build(order, deadline):
    """Greedy build from a given order."""
    current = []
    for x in order:
        if time.time() > deadline:
            break
        candidate = current + [x]
        if _verify(candidate):
            current = candidate
    return current

def _collision_remove(universe, heuristic, deadline):
    """Start with full set, remove elements causing collisions."""
    current = list(universe)
    while time.time() < deadline:
        col = _find_collision(current)
        if col is None:
            break
        mask1, mask2 = col
        k = len(current)
        sym_diff = []
        for i in range(k):
            bit = 1 << i
            if (mask1 & bit) != (mask2 & bit):
                sym_diff.append(i)
        if heuristic == 'largest':
            to_remove = max(sym_diff, key=lambda i: current[i])
        elif heuristic == 'smallest':
            to_remove = min(sym_diff, key=lambda i: current[i])
        elif heuristic == 'random':
            to_remove = random.choice(sym_diff)
        else:
            # Remove element with largest scaled weight (most impactful)
            to_remove = max(sym_diff, key=lambda i: _LCM // current[i])
        current.remove(current[to_remove])
    return current

def _try_add(best, remaining, deadline):
    """Try to add back elements one by one."""
    current = list(best)
    for x in remaining:
        if time.time() > deadline:
            break
        candidate = current + [x]
        if _verify(candidate):
            current = candidate
    return current

def _local_search(best, universe, deadline):
    """Local search: try swaps and additions."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        in_set = set(best)
        remaining = [x for x in universe if x not in in_set]

        # Try adding elements
        for x in remaining:
            if time.time() > deadline:
                break
            candidate = best + [x]
            if _verify(candidate):
                best = candidate
                improved = True
                break

        # Try single swaps that enable growth
        if not improved:
            in_set = set(best)
            remaining = [x for x in universe if x not in in_set]
            for i, a in enumerate(best):
                if time.time() > deadline:
                    break
                for b in remaining:
                    candidate = best[:]
                    candidate[i] = b
                    if _verify(candidate):
                        new_in = set(candidate)
                        new_remaining = [x for x in universe if x not in new_in]
                        for c in new_remaining:
                            if time.time() > deadline:
                                break
                            if _verify(candidate + [c]):
                                best = candidate + [c]
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
    return best

def _sa_polish(best, universe, deadline):
    """Simulated annealing polish: swap elements to improve min-gap surrogate.
    Accept moves that maintain dissociation; among those, prefer larger min-gap.
    Also allow temporary non-dissociated states with temperature-based acceptance."""
    current = list(best)
    current_set = set(current)
    current_dissociated = _verify(current)
    current_gap = _min_gap(current) if current_dissociated else 0
    best_found = list(current)
    best_gap = current_gap if current_dissociated else -1

    rng = random.Random(12345)
    t0 = time.time()
    # Temperature schedule: start high, decay
    T_start = 500000.0
    T_end = 1000.0
    max_iters = 50000

    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Temperature decay
        progress = min(1.0, it / max_iters)
        T = T_start * (T_end / T_start) ** progress

        if rng.random() < 0.3:
            # Try to add an element
            not_in = [x for x in universe if x not in current_set]
            if not not_in:
                continue
            x = rng.choice(not_in)
            candidate = current + [x]
        else:
            # Swap: remove one, add one
            if len(current) < 2:
                continue
            i = rng.randrange(len(current))
            a = current[i]
            not_in = [x for x in universe if x not in current_set]
            if not not_in:
                continue
            b = rng.choice(not_in)
            candidate = current[:]
            candidate[i] = b

        candidate_set = set(candidate)
        candidate_dissociated = _verify(candidate)
        candidate_gap = _min_gap(candidate) if candidate_dissociated else 0

        # Acceptance: prefer larger cardinality, then larger gap
        if len(candidate) > len(current):
            accept = True
        elif len(candidate) < len(current):
            # Only accept if it's a swap (same size) or if gap improves significantly
            accept = False
        else:
            # Same size: compare gaps
            if candidate_dissociated and not current_dissociated:
                accept = True
            elif not candidate_dissociated and current_dissociated:
                # Reject unless temperature allows
                delta = -1000000  # penalty for losing dissociation
                accept = rng.random() < min(1.0, delta / max(T, 1))
            else:
                # Both dissociated or both not
                delta = candidate_gap - current_gap
                if delta >= 0:
                    accept = True
                else:
                    accept = rng.random() < min(1.0, delta / max(T, 1))

        if accept:
            current = candidate
            current_set = candidate_set
            current_dissociated = candidate_dissociated
            current_gap = candidate_gap
            # Track best
            if current_dissociated and len(current) > len(best_found):
                best_found = list(current)
                best_gap = current_gap
            elif current_dissociated and len(current) == len(best_found) and current_gap > best_gap:
                best_found = list(current)
                best_gap = current_gap

    # After SA, try to add more elements
    best_found = _try_add(best_found, [x for x in universe if x not in set(best_found)], deadline)
    return best_found

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill

    universe = list(range(1, 23))

    best = []
    best_size = 0

    # Phase 1: Collision-driven removal with different heuristics
    heuristics = ['largest', 'smallest', 'random', 'scaled_largest']
    random.seed(42)
    for h in heuristics:
        if time.time() > deadline - 30:
            break
        result = _collision_remove(universe, h, deadline - 30)
        removed = [x for x in universe if x not in result]
        result = _try_add(result, removed, deadline - 25)
        result = _local_search(result, universe, deadline - 20)
        if len(result) > best_size:
            best_size = len(result)
            best = result

    # Phase 2: Greedy build with various orderings
    orderings = [
        sorted(universe, key=lambda x: -(_LCM // x)),  # largest scaled first
        sorted(universe, key=lambda x: _LCM // x),      # smallest scaled first
        sorted(universe, reverse=True),                  # largest x first
        sorted(universe),                                # smallest x first
    ]
    random.seed(43)
    for _ in range(15):
        orderings.append(universe[:])
        random.shuffle(orderings[-1])

    for order in orderings:
        if time.time() > deadline - 20:
            break
        candidate = _greedy_build(order, deadline - 20)
        candidate = _local_search(candidate, universe, deadline - 15)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best = candidate

    # Phase 3: SA polish on the best found
    if time.time() < deadline - 5:
        polished = _sa_polish(best, universe, deadline - 3)
        if len(polished) > best_size:
            best = polished
            best_size = len(polished)

    # Phase 4: Final local search and addition
    if time.time() < deadline - 2:
        best = _local_search(best, universe, deadline - 1)
        in_set = set(best)
        remaining = [x for x in universe if x not in in_set]
        for x in remaining:
            if time.time() > deadline - 1:
                break
            candidate = best + [x]
            if _verify(candidate):
                best = candidate

    # Ensure nonempty
    if not best:
        best = [1]

    return best
