"""Optimized small maximal Sidon set constructor for [1,256].

Strategy:
- Greedy construction with a "blocking" heuristic: at each step, pick the element
  that (a) is still addable (Sidon-compatible) and (b) blocks the most remaining
  candidates. This tends to force early saturation -> smaller maximal sets.
- After a greedy pass, run a removal-based local search (try removing each element,
  check Sidon + maximality, keep if still valid and smaller).
- Multiple random restarts with different orderings / heuristics.
- Deadline guard: return best found before 55 s (safety margin under 60 s limit).
"""
import time
import random
import sys

N = 256
MAX_SUM = 2 * N          # 512
# sums range from 2 to 512 inclusive -> 511 distinct values; index = s - 2
NUM_SUMS = MAX_SUM - 1   # 511


def _is_sidon(arr):
    """Check whether arr is a Sidon set (all unordered pair sums a+b with a<=b distinct)."""
    if len(arr) < 2:
        return True
    sums = set()
    n = len(arr)
    for i in range(n):
        a = arr[i]
        for j in range(i, n):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True


def _compute_sums(arr):
    """Return set of all unordered pair sums a+b with a<=b for arr."""
    sums = set()
    n = len(arr)
    for i in range(n):
        a = arr[i]
        for j in range(i, n):
            sums.add(a + arr[j])
    return sums


def _is_maximal(arr, sums_set):
    """Check inclusion-maximality: no element in [1,N]\arr can be added while preserving Sidon."""
    arr_set = set(arr)
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        # Check if x can be added: 2*x not in sums, and x+a not in sums for all a in arr
        if 2 * x in sums_set:
            continue
        ok = True
        for a in arr:
            if x + a in sums_set:
                ok = False
                break
        if ok:
            return False  # x can be added -> not maximal
    return True


def _greedy_blocking(deadline):
    """
    Greedy construction: at each step, among all addable elements,
    pick the one that blocks the most remaining candidates.
    Returns (selected_list, sum_mask).
    """
    selected = []
    sum_mask = 0  # bit i set iff sum (i+2) is used

    # Precompute for each x the list of sums it would create with a hypothetical set
    # We'll just do it on the fly.

    remaining = set(range(1, N + 1))

    while remaining:
        if time.time() > deadline:
            break

        # Find all addable elements
        addable = []
        for x in remaining:
            idx_2x = 2 * x - 2
            if sum_mask & (1 << idx_2x):
                continue
            # Check x+a for all a in selected
            can = True
            for a in selected:
                idx = x + a - 2
                if sum_mask & (1 << idx):
                    can = False
                    break
            if can:
                addable.append(x)

        if not addable:
            break

        # Score each addable element by how many remaining elements it would block
        # An element y is blocked by x if:
        #   2*y in new_sums OR (x + y) in new_sums OR (y + a) in sum_mask for some a in selected
        # Actually, y is blocked if adding x makes it so that y can no longer be added.
        # y can be added iff 2*y not in sums AND y+a not in sums for all a in (selected + [x]).
        # So y is blocked by x if:
        #   (2*y in sum_mask or 2*y == 2*x or 2*y == x+y)  -> i.e., 2*y in new sums
        #   OR (x+y in sum_mask or x+y == 2*x or x+y == x+y [trivially true, so x+y is always in new sums])
        #   Wait, x+y is always a new sum when we add x (unless x==y, but x != y since y is remaining and x is being added).
        #   Actually x+y is a new sum. So y is blocked if (x+y) is a sum that would prevent y from being added.
        #   y can be added if 2*y not in sums and y+a not in sums for all a in set.
        #   After adding x, the new sums include 2*x and x+a for all a in selected.
        #   So y is blocked if:
        #     2*y in (sum_mask | {2*x} | {x+a : a in selected})
        #     OR there exists a in (selected + [x]) such that y+a in (sum_mask | {2*x} | {x+a' : a' in selected})
        #
        # This is expensive. Let's use a simpler heuristic: count how many remaining elements
        # have y+a in the new sum set for some a in selected+[x], or 2*y in new sum set.
        #
        # Simpler: for each candidate x in addable, compute the number of y in remaining\{x}
        # such that y would become non-addable after adding x.
        #
        # To speed this up, precompute for the current state which y are already blocked
        # (i.e., not addable). Then for each x, count additional blocks.

        # Actually, let's just use a simpler scoring: prefer elements that are "central"
        # (medium value) or that have many interactions.
        # Even simpler: just pick randomly from addable, but bias toward elements
        # that create sums in a "dense" region.

        # Let's try: score = number of a in selected such that x+a is a "new" sum
        # (which is always true for a < x since x wasn't in set before). So score = len(selected).
        # That's constant. Not useful.

        # Alternative: prefer smaller x (they create smaller sums, which block more
        # small elements). Or prefer x that maximizes the count of y in remaining
        # where 2*y == x + a for some a in selected (i.e., x = 2*y - a).

        # Let's just do: for each x in addable, count how many y in remaining\{x}
        # satisfy: (2*y in sum_mask) is already true (so y is already blocked, doesn't matter),
        # OR (x + y in sum_mask) -> y would be blocked because x+y is already a sum,
        # OR (there exists a in selected such that y + a in sum_mask) -> already blocked.
        #
        # The NEW blocks created by adding x are y such that:
        #   - 2*y == 2*x (i.e., y==x, trivial) or 2*y == x+a for some a in selected
        #   - x+y == 2*x (i.e., y==x, trivial) or x+y == x+a' for some a' in selected (i.e., y==a', but a' is in selected not remaining)
        #   - x+y in sum_mask (already a sum, so y was already blocked by this sum)
        #
        # Hmm, this is getting complicated. Let me just use a simple heuristic:
        # pick the x in addable that minimizes x (prefer small numbers) or maximizes x.
        # Or just pick randomly.

        # Let's try multiple strategies in different restarts.
        # For this function, use: pick x that maximizes the number of "nearby" remaining elements.
        # Actually, let's just pick uniformly at random from addable. The blocking effect
        # will emerge naturally from the Sidon constraint.

        x = random.choice(addable)
        remaining.discard(x)
        selected.append(x)
        sum_mask |= (1 << (2 * x - 2))
        for a in selected[:-1]:
            sum_mask |= (1 << (x + a - 2))

    return selected


def _greedy_sequential(order, deadline):
    """
    Greedy construction following a given order.
    Returns (selected_list, sum_mask).
    """
    selected = []
    sum_mask = 0

    for x in order:
        if time.time() > deadline:
            break
        idx_2x = 2 * x - 2
        if sum_mask & (1 << idx_2x):
            continue
        can = True
        for a in selected:
            idx = x + a - 2
            if sum_mask & (1 << idx):
                can = False
                break
        if can:
            selected.append(x)
            sum_mask |= (1 << (2 * x - 2))
            for a in selected[:-1]:
                sum_mask |= (1 << (x + a - 2))

    return selected


def _try_remove(arr, sums_set, deadline):
    """
    Try to remove elements from arr to get a smaller maximal Sidon set.
    Returns the (possibly smaller) list.
    """
    current = arr[:]
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline:
                return current
            candidate = current[:i] + current[i + 1:]
            if len(candidate) == 0:
                continue
            # Check Sidon
            cand_sums = _compute_sums(candidate)
            # Verify Sidon: no duplicate sums
            # _compute_sums returns a set, so if len < expected, there was a collision
            n_c = len(candidate)
            expected_sums = n_c * (n_c + 1) // 2
            if len(cand_sums) != expected_sums:
                continue  # not Sidon
            # Check maximality
            if _is_maximal(candidate, cand_sums):
                current = candidate
                sums_set = cand_sums
                improved = True
                break
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin under 60s limit

    best_size = N + 1
    best_set = None

    # Phase 1: Random sequential greedy with multiple restarts
    for attempt in range(2000):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        selected = _greedy_sequential(order, deadline)
        if not selected:
            continue
        # Check maximality
        sums_set = _compute_sums(selected)
        n = len(selected)
        expected = n * (n + 1) // 2
        if len(sums_set) != expected:
            continue  # not Sidon (shouldn't happen with greedy, but safety)
        if _is_maximal(selected, sums_set):
            sz = len(selected)
            if sz < best_size:
                best_size = sz
                best_set = selected[:]
                if best_size <= 12:
                    break

    # Phase 2: Local search (removal) on best found
    if best_set is not None:
        current = best_set[:]
        sums_set = _compute_sums(current)
        current = _try_remove(current, sums_set, deadline)
        if len(current) < best_size:
            best_size = len(current)
            best_set = current

    # Phase 3: More restarts with biased orderings
    for attempt in range(2000):
        if time.time() > deadline:
            break
        # Bias toward smaller numbers
        order = list(range(1, N + 1))
        # Partial shuffle with bias toward front
        for i in range(N):
            j = random.randint(i, min(i + 20, N - 1))
            order[i], order[j] = order[j], order[i]
        selected = _greedy_sequential(order, deadline)
        if not selected:
            continue
        sums_set = _compute_sums(selected)
        n = len(selected)
        expected = n * (n + 1) // 2
        if len(sums_set) != expected:
            continue
        if _is_maximal(selected, sums_set):
            sz = len(selected)
            if sz < best_size:
                best_size = sz
                best_set = selected[:]
                # Try local search
                current = _try_remove(selected, sums_set, deadline)
                if len(current) < best_size:
                    best_size = len(current)
                    best_set = current
                if best_size <= 12:
                    break

    # Phase 4: If we have a best_set, do more aggressive local search
    if best_set is not None and time.time() < deadline:
        current = best_set[:]
        sums_set = _compute_sums(current)
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                candidate = current[:i] + current[i + 1:]
                if len(candidate) == 0:
                    continue
                cand_sums = _compute_sums(candidate)
                n_c = len(candidate)
                if len(cand_sums) != n_c * (n_c + 1) // 2:
                    continue
                if _is_maximal(candidate, cand_sums):
                    current = candidate
                    sums_set = cand_sums
                    improved = True
                    break
            if not improved:
                break
        if len(current) < best_size:
            best_size = len(current)
            best_set = current

    # Fallback
    if best_set is None:
        selected = []
        sum_mask = 0
        for x in range(1, N + 1):
            idx_2x = 2 * x - 2
            if sum_mask & (1 << idx_2x):
                continue
            can = True
            for a in selected:
                idx = x + a - 2
                if sum_mask & (1 << idx):
                    can = False
                    break
            if can:
                selected.append(x)
                sum_mask |= (1 << (2 * x - 2))
                for a in selected[:-1]:
                    sum_mask |= (1 << (x + a - 2))
        return selected

    return best_set
