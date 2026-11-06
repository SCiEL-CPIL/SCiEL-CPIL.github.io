"""Rewritten program for B2[2] set construction on {1,...,1000}."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 1000
MAX_SUM = 2 * N  # 2000


def _count_sums(chosen):
    """Return an array of length MAX_SUM+1 with counts of pairwise sums a+b (a<=b)."""
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    arr = np.array(chosen, dtype=np.int32)
    n = len(arr)
    if n == 0:
        return counts
    i_idx, j_idx = np.triu_indices(n, 0)
    sums = arr[i_idx] + arr[j_idx]
    np.add.at(counts, sums, 1)
    return counts


def _can_add(chosen_set, counts, x):
    """Check if x can be added to the set without violating B2[2]."""
    for y in chosen_set:
        s = x + y
        if s <= MAX_SUM and counts[s] >= 2:
            return False
    if 2 * x <= MAX_SUM and counts[2 * x] >= 2:
        return False
    return True


def _add_element(chosen, chosen_set, counts, x):
    """Add x to chosen, updating counts in-place."""
    chosen.append(x)
    chosen_set.add(x)
    for y in chosen[:-1]:
        s = x + y
        counts[s] += 1
    counts[2 * x] += 1


def _remove_element(chosen, chosen_set, counts, x):
    """Remove x from chosen, updating counts in-place."""
    chosen.remove(x)
    chosen_set.discard(x)
    for y in chosen:
        s = x + y
        counts[s] -= 1
    counts[2 * x] -= 1


def _greedy_asc():
    """Simple ascending greedy."""
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    for x in range(1, N + 1):
        if _can_add(chosen_set, counts, x):
            _add_element(chosen, chosen_set, counts, x)
    return chosen


def _greedy_desc():
    """Simple descending greedy."""
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    for x in range(N, 0, -1):
        if _can_add(chosen_set, counts, x):
            _add_element(chosen, chosen_set, counts, x)
    return chosen


def _greedy_random(seed=0):
    """Greedy with randomized order."""
    random.seed(seed)
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    order = list(range(1, N + 1))
    random.shuffle(order)
    for x in order:
        if _can_add(chosen_set, counts, x):
            _add_element(chosen, chosen_set, counts, x)
    return chosen


def _greedy_density_aware(seed=0):
    """
    Density-aware greedy: at each step, pick the candidate that creates the fewest
    'near-conflict' sums (sums that would go from count 1 to 2).
    """
    random.seed(seed)
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)

    while candidates:
        best_x = None
        best_cost = float('inf')
        best_idx = -1

        # Evaluate a sample of candidates
        sample_size = min(len(candidates), 150)
        sample_indices = random.sample(range(len(candidates)), sample_size)

        for idx in sample_indices:
            x = candidates[idx]
            # Cost: number of sums that would reach count 2 (i.e., currently at 1)
            cost = 0
            for y in chosen_set:
                s = x + y
                if s <= MAX_SUM and counts[s] == 1:
                    cost += 1
            if 2 * x <= MAX_SUM and counts[2 * x] == 1:
                cost += 1

            # Check validity
            valid = True
            for y in chosen_set:
                s = x + y
                if s <= MAX_SUM and counts[s] >= 2:
                    valid = False
                    break
            if valid and 2 * x <= MAX_SUM and counts[2 * x] >= 2:
                valid = False

            if not valid:
                continue

            if cost < best_cost:
                best_cost = cost
                best_x = x
                best_idx = idx

            if cost == 0:
                break

        # If no zero-cost found in sample, scan all
        if best_cost > 0:
            for idx in range(len(candidates)):
                x = candidates[idx]
                valid = True
                for y in chosen_set:
                    s = x + y
                    if s <= MAX_SUM and counts[s] >= 2:
                        valid = False
                        break
                if valid and 2 * x <= MAX_SUM and counts[2 * x] >= 2:
                    valid = False
                if not valid:
                    continue
                cost = 0
                for y in chosen_set:
                    s = x + y
                    if s <= MAX_SUM and counts[s] == 1:
                        cost += 1
                if 2 * x <= MAX_SUM and counts[2 * x] == 1:
                    cost += 1
                if cost < best_cost:
                    best_cost = cost
                    best_x = x
                    best_idx = idx
                if cost == 0:
                    break

        if best_x is None:
            break

        _add_element(chosen, chosen_set, counts, best_x)
        candidates.pop(best_idx)

    return chosen


def _local_search(chosen, max_iters=300, deadline=None):
    """
    Local search: repeatedly try to swap an element for a better one.
    Uses a scoring heuristic to prefer additions that create fewer future conflicts.
    """
    if deadline is None:
        deadline = time.time() + 60

    chosen = sorted(chosen)
    counts = _count_sums(chosen)
    chosen_set = set(chosen)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > deadline:
            break

        # Try removing each element and finding a better replacement
        for i in range(len(chosen)):
            x_remove = chosen[i]

            # Temporarily remove
            temp_counts = counts.copy()
            for y in chosen:
                if y == x_remove:
                    continue
                s = x_remove + y
                temp_counts[s] -= 1
            temp_counts[2 * x_remove] -= 1

            temp_chosen_set = chosen_set - {x_remove}

            # Find best addition
            best_add = None
            best_score = -1

            # Generate candidates: all numbers not in temp_chosen_set
            # Sample to keep it fast
            candidates = [x for x in range(1, N + 1) if x not in temp_chosen_set]
            random.shuffle(candidates)
            sample = candidates[:80]

            for x_add in sample:
                # Check validity
                valid = True
                for y in temp_chosen_set:
                    s = x_add + y
                    if s <= MAX_SUM and temp_counts[s] >= 2:
                        valid = False
                        break
                if valid and 2 * x_add <= MAX_SUM and temp_counts[2 * x_add] >= 2:
                    valid = False

                if not valid:
                    continue

                # Score: prefer additions that create fewer sums at count 1
                # (i.e., less likely to block future additions)
                score = 0
                for y in temp_chosen_set:
                    s = x_add + y
                    if s <= MAX_SUM and temp_counts[s] == 1:
                        score += 1
                if 2 * x_add <= MAX_SUM and temp_counts[2 * x_add] == 1:
                    score += 1

                if score < best_score:
                    best_score = score
                    best_add = x_add

            if best_add is not None and best_score < len(temp_chosen_set):
                # Perform the swap
                _remove_element(chosen, chosen_set, counts, x_remove)
                _add_element(chosen, chosen_set, counts, best_add)
                chosen = sorted(chosen)
                counts = _count_sums(chosen)
                improved = True
                break

    return chosen


def _try_add_to_set(chosen, chosen_set, counts, x):
    """Try to add x; return True if added."""
    if _can_add(chosen_set, counts, x):
        _add_element(chosen, chosen_set, counts, x)
        return True
    return False


def _saturating_greedy(seed=0):
    """
    Saturating greedy: repeatedly scan all remaining candidates and add any valid one.
    Continue until no more can be added.
    """
    random.seed(seed)
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()
    remaining = set(range(1, N + 1))

    progress = True
    while progress and remaining:
        progress = False
        # Shuffle for variety
        order = list(remaining)
        random.shuffle(order)
        for x in order:
            if x not in remaining:
                continue
            if _can_add(chosen_set, counts, x):
                _add_element(chosen, chosen_set, counts, x)
                remaining.discard(x)
                progress = True

    return chosen


def _conflict_graph_greedy(seed=0):
    """
    Model as maximum independent set on conflict graph.
    Two elements conflict if adding both would create a sum with >2 representations.
    Use greedy coloring-like heuristic with randomized ordering.
    """
    random.seed(seed)
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    chosen = []
    chosen_set = set()

    # Randomized order
    order = list(range(1, N + 1))
    random.shuffle(order)

    for x in order:
        if _can_add(chosen_set, counts, x):
            _add_element(chosen, chosen_set, counts, x)

    return chosen


def _polish(chosen, deadline=None):
    """
    Final polish: try to add any remaining valid elements, then do a few rounds
    of swap-based local search.
    """
    if deadline is None:
        deadline = time.time() + 30

    chosen = sorted(chosen)
    counts = _count_sums(chosen)
    chosen_set = set(chosen)

    # Try adding any missing valid elements
    for x in range(1, N + 1):
        if x in chosen_set:
            continue
        if _can_add(chosen_set, counts, x):
            _add_element(chosen, chosen_set, counts, x)

    # A few rounds of local search
    if time.time() < deadline - 5:
        chosen = _local_search(chosen, max_iters=50, deadline=deadline)

    return chosen


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    best_set = []
    best_size = 0

    # Strategy 1: Ascending greedy
    set1 = _greedy_asc()
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1

    if time.time() > deadline - 100:
        return _polish(best_set, deadline)

    # Strategy 2: Descending greedy
    set2 = _greedy_desc()
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2

    if time.time() > deadline - 90:
        return _polish(best_set, deadline)

    # Strategy 3: Multiple random-order greedies
    for trial in range(10):
        if time.time() > deadline - 80:
            break
        s = _greedy_random(seed=trial * 7 + 1)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 70:
        return _polish(best_set, deadline)

    # Strategy 4: Density-aware greedy
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        s = _greedy_density_aware(seed=trial * 13 + 3)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 50:
        return _polish(best_set, deadline)

    # Strategy 5: Saturating greedy
    for trial in range(3):
        if time.time() > deadline - 40:
            break
        s = _saturating_greedy(seed=trial * 17 + 5)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 30:
        return _polish(best_set, deadline)

    # Strategy 6: Conflict graph greedy
    for trial in range(3):
        if time.time() > deadline - 20:
            break
        s = _conflict_graph_greedy(seed=trial * 23 + 7)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Final polish on the best set
    if time.time() < deadline - 5:
        polished = _polish(best_set, deadline)
        if len(polished) > len(best_set):
            best_set = polished

    return best_set
# EVOLVE-BLOCK-END
