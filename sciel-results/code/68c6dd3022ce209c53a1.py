import time
import random
import numpy as np

N = 128
TARGET_LO = 128
TARGET_HI = 256
NUM_TARGETS = 129  # targets 128..256 inclusive

_START_TIME = time.time()
DEADLINE = 85.0  # Safety margin before 90s kill


def _elapsed():
    return time.time() - _START_TIME


def _pair_counts(A):
    """Compute representation counts for each target in [128,256]."""
    counts = np.zeros(NUM_TARGETS, dtype=np.int32)
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            if TARGET_LO <= s <= TARGET_HI:
                counts[s - TARGET_LO] += 1
    return counts


def _is_valid(counts):
    return np.min(counts) > 0


def _get_peak(counts):
    return np.max(counts)


def _variance_penalty(counts):
    """Return variance of counts as a small penalty for imbalance."""
    return float(np.var(counts.astype(np.float64)))


def _score(counts):
    """Lower is better. Peak + variance penalty, or infinity if invalid."""
    if not _is_valid(counts):
        return float('inf')
    return _get_peak(counts) + 0.05 * _variance_penalty(counts)


def _try_swap(A, A_set, counts, i, x_new):
    """
    Try replacing A[i] with x_new.
    Returns (new_counts, new_score) or None if invalid.
    """
    ai = A[i]
    if x_new == ai or x_new in A_set:
        return None

    n = len(A)
    new_counts = counts.copy()

    # Remove contributions of ai
    s = ai + ai
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] -= 1

    for j in range(n):
        if j == i:
            continue
        aj = A[j]
        s = ai + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] -= 1

    # Add contributions of x_new
    s = x_new + x_new
    if TARGET_LO <= s <= TARGET_HI:
        new_counts[s - TARGET_LO] += 1

    for j in range(n):
        if j == i:
            continue
        aj = A[j]
        s = x_new + aj
        if TARGET_LO <= s <= TARGET_HI:
            new_counts[s - TARGET_LO] += 1

    # Check validity
    if np.min(new_counts) <= 0:
        return None

    new_score = _score(new_counts)
    return (new_counts, new_score)


def _local_search(A, max_time=80.0):
    """
    Perform swap-based local search to minimize score (peak + variance).
    Returns (best_A, best_peak).
    """
    A = sorted(A)
    A_set = set(A)
    counts = _pair_counts(A)

    if not _is_valid(counts):
        return (A, float('inf'))

    current_score = _score(counts)
    current_peak = _get_peak(counts)
    best_A = list(A)
    best_peak = current_peak
    best_score = current_score

    n = len(A)
    improved = True
    stall_count = 0
    max_stall = 50

    while improved and _elapsed() < max_time:
        improved = False
        stall_count += 1

        indices = list(range(n))
        random.shuffle(indices)

        for i in indices:
            if _elapsed() > max_time:
                break
            ai = A[i]

            candidates = [x for x in range(0, N + 1) if x not in A_set and x != ai]
            random.shuffle(candidates)

            for x_new in candidates:
                result = _try_swap(A, A_set, counts, i, x_new)
                if result is None:
                    continue
                new_counts, new_score = result

                if new_score < current_score - 1e-9:
                    # Accept
                    A[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    A.sort()
                    counts = new_counts
                    current_score = new_score
                    current_peak = _get_peak(counts)
                    improved = True
                    stall_count = 0
                    if current_peak < best_peak or (current_peak == best_peak and current_score < best_score):
                        best_peak = current_peak
                        best_score = current_score
                        best_A = list(A)
                    break  # Restart from beginning after a move

        if not improved and stall_count >= max_stall:
            break

    return (best_A, best_peak)


def _balance_polish(A, max_time=80.0):
    """
    Polish: try to reduce variance of counts without increasing peak.
    This can help escape local minima by flattening the distribution.
    """
    A = sorted(A)
    A_set = set(A)
    counts = _pair_counts(A)

    if not _is_valid(counts):
        return (A, float('inf'))

    current_peak = _get_peak(counts)
    current_var = _variance_penalty(counts)
    best_A = list(A)
    best_peak = current_peak
    best_var = current_var

    n = len(A)
    improved = True
    stall_count = 0
    max_stall = 30

    while improved and _elapsed() < max_time:
        improved = False
        stall_count += 1

        indices = list(range(n))
        random.shuffle(indices)

        for i in indices:
            if _elapsed() > max_time:
                break
            ai = A[i]

            candidates = [x for x in range(0, N + 1) if x not in A_set and x != ai]
            random.shuffle(candidates)

            for x_new in candidates:
                result = _try_swap(A, A_set, counts, i, x_new)
                if result is None:
                    continue
                new_counts, new_score = result

                new_peak = _get_peak(new_counts)
                new_var = _variance_penalty(new_counts)

                # Accept if peak decreases, or peak same and variance decreases
                if new_peak < current_peak or (new_peak == current_peak and new_var < best_var - 1e-9):
                    A[i] = x_new
                    A_set.discard(ai)
                    A_set.add(x_new)
                    A.sort()
                    counts = new_counts
                    current_peak = new_peak
                    current_var = new_var
                    improved = True
                    stall_count = 0

                    if current_peak < best_peak or (current_peak == best_peak and new_var < best_var - 1e-9):
                        best_peak = current_peak
                        best_var = new_var
                        best_A = list(A)
                    break

        if not improved and stall_count >= max_stall:
            break

    return (best_A, best_peak)


def _simulated_annealing(A, max_time=80.0):
    """
    Simulated annealing to escape local optima.
    Allows worsening moves with probability based on 'temperature'.
    """
    A = sorted(A)
    A_set = set(A)
    counts = _pair_counts(A)

    if not _is_valid(counts):
        return (A, float('inf'))

    current_score = _score(counts)
    current_peak = _get_peak(counts)
    best_A = list(A)
    best_peak = current_peak
    best_score = current_score

    n = len(A)
    T = 2.0  # Initial temperature
    cooling = 0.995

    while _elapsed() < max_time:
        T *= cooling
        if T < 0.01:
            break

        i = random.randint(0, n - 1)
        ai = A[i]

        candidates = [x for x in range(0, N + 1) if x not in A_set and x != ai]
        if not candidates:
            continue
        x_new = random.choice(candidates)

        result = _try_swap(A, A_set, counts, i, x_new)
        if result is None:
            continue
        new_counts, new_score = result

        delta = new_score - current_score
        if delta <= 0:
            # Accept improvement or equal
            A[i] = x_new
            A_set.discard(ai)
            A_set.add(x_new)
            A.sort()
            counts = new_counts
            current_score = new_score
            current_peak = _get_peak(counts)

            if current_peak < best_peak or (current_peak == best_peak and current_score < best_score):
                best_peak = current_peak
                best_score = current_score
                best_A = list(A)
        else:
            # Accept worsening with probability exp(-delta / T)
            if random.random() < np.exp(-delta / T):
                A[i] = x_new
                A_set.discard(ai)
                A_set.add(x_new)
                A.sort()
                counts = new_counts
                current_score = new_score
                current_peak = _get_peak(counts)

    return (best_A, best_peak)


def _greedy_build(rng, max_time=80.0):
    """
    Build a set greedily. Start with empty, add elements that cover targets.
    """
    A = []
    A_set = set()
    counts = np.zeros(NUM_TARGETS, dtype=np.int32)

    # Initial anchors to help cover extremes
    initial = [0, 128, 64]
    for x in initial:
        if x not in A_set:
            A.append(x)
            A_set.add(x)
    A.sort()
    counts = _pair_counts(A)

    while _elapsed() < max_time:
        if _is_valid(counts):
            break

        # Find uncovered targets
        uncovered_indices = np.where(counts == 0)[0]
        if len(uncovered_indices) == 0:
            break

        # Pick the target with the "hardest" coverage
        target_idx = rng.choice(uncovered_indices)
        target_val = TARGET_LO + target_idx

        best_x = None
        best_gain = -1

        for x in range(0, N + 1):
            if x in A_set:
                continue

            # Calculate how many new targets this x would cover
            gain = 0
            # Pairs with existing A
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI:
                    if counts[s - TARGET_LO] == 0:
                        gain += 1
            # Pair with itself
            s = x + x
            if TARGET_LO <= s <= TARGET_HI:
                if counts[s - TARGET_LO] == 0:
                    gain += 1

            if gain > best_gain:
                best_gain = gain
                best_x = x

        if best_x is None or best_gain <= 0:
            break

        A.append(best_x)
        A_set.add(best_x)
        A.sort()
        counts = _pair_counts(A)

    return A, counts


def _greedy_balanced_build(rng, max_time=80.0):
    """
    Build a set greedily to cover all targets while balancing representation counts.
    Returns (A_sorted, counts).
    """
    A = []
    A_set = set()
    counts = np.zeros(NUM_TARGETS, dtype=np.int32)

    # Start with a few random elements to get some coverage
    remaining = list(range(0, N + 1))
    rng.shuffle(remaining)

    for _ in range(10):
        if remaining:
            x = remaining.pop()
            A.append(x)
            A_set.add(x)

    A.sort()
    counts = _pair_counts(A)

    # Now greedily add elements that help balance the distribution
    max_iter = 100
    for _ in range(max_iter):
        if _elapsed() > max_time:
            break
        if _is_valid(counts):
            break

        # Find the target with minimum count
        min_count = np.min(counts)
        min_indices = np.where(counts == min_count)[0]
        min_target_idx = rng.choice(min_indices)

        # Find a candidate x in [0,128] not in A that helps cover low-count targets
        best_x = None
        best_gain = -1

        for x in remaining:
            if x in A_set:
                continue
            # How many new pairs does x create that hit low-count targets?
            gain = 0
            for y in A:
                s = x + y
                if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] <= 2:
                    gain += 1
            s = x + x
            if TARGET_LO <= s <= TARGET_HI and counts[s - TARGET_LO] <= 2:
                gain += 1

            if gain > best_gain:
                best_gain = gain
                best_x = x

        if best_x is None or best_gain <= 0:
            if remaining:
                best_x = remaining.pop()
            else:
                break

        if best_x in remaining:
            remaining.remove(best_x)

        A.append(best_x)
        A_set.add(best_x)
        A.sort()
        counts = _pair_counts(A)

    # Try to reduce size by removing redundant elements
    changed = True
    while changed and len(A) > 15:
        changed = False
        for i in range(len(A)):
            ai = A[i]
            new_counts = counts.copy()
            n = len(A)
            for j in range(n):
                if j == i:
                    continue
                aj = A[j]
                s = ai + aj
                if TARGET_LO <= s <= TARGET_HI:
                    new_counts[s - TARGET_LO] -= 1
            s = ai + ai
            if TARGET_LO <= s <= TARGET_HI:
                new_counts[s - TARGET_LO] -= 1

            if _is_valid(new_counts):
                A.pop(i)
                A_set.discard(ai)
                counts = new_counts
                changed = True
                break

    return A, counts


def construct():
    global _START_TIME
    _START_TIME = time.time()

    rng = random.Random(42)

    best_A = None
    best_peak = float('inf')

    # Strategy:
    # 1. Generate initial sets of various sizes using random sampling and greedy methods.
    # 2. Apply local search, balance polish, and simulated annealing to each.
    # 3. Keep the best.

    sizes_to_try = [15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 35, 40]

    for size in sizes_to_try:
        if _elapsed() > DEADLINE - 5:
            break

        # Multiple restarts for each size
        restarts = 3

        for r in range(restarts):
            if _elapsed() > DEADLINE - 2:
                break

            # Generate initial A
            if r == 0:
                # Greedy balanced start
                A_init, _ = _greedy_balanced_build(rng, max_time=DEADLINE - 10)
                # Adjust size if greedy produced a different size
                if len(A_init) < size:
                    A_set = set(A_init)
                    while len(A_init) < size:
                        x = rng.randint(0, N)
                        if x not in A_set:
                            A_init.append(x)
                            A_set.add(x)
                elif len(A_init) > size:
                    A_init = rng.sample(A_init, size)
            elif r == 1:
                # Greedy start
                A_init, _ = _greedy_build(rng, max_time=DEADLINE - 10)
                if len(A_init) < size:
                    A_set = set(A_init)
                    while len(A_init) < size:
                        x = rng.randint(0, N)
                        if x not in A_set:
                            A_init.append(x)
                            A_set.add(x)
                elif len(A_init) > size:
                    A_init = rng.sample(A_init, size)
            else:
                # Random start
                A_init = rng.sample(range(0, N + 1), size)

            A_init = sorted(A_init)

            # Check validity
            counts = _pair_counts(A_init)
            if not _is_valid(counts):
                # Try to fix coverage by adding elements
                A_fixed, _ = _greedy_build(rng, max_time=DEADLINE - 10)
                A_temp = list(A_init)
                A_set = set(A_temp)
                counts_temp = counts
                for _ in range(10):  # Max 10 additions
                    if _is_valid(counts_temp):
                        break
                    uncovered = np.where(counts_temp == 0)[0]
                    if len(uncovered) == 0:
                        break
                    target_idx = rng.choice(uncovered)
                    best_x = None
                    best_gain = -1
                    for x in range(0, N + 1):
                        if x in A_set:
                            continue
                        gain = 0
                        for y in A_temp:
                            s = x + y
                            if TARGET_LO <= s <= TARGET_HI and counts_temp[s - TARGET_LO] == 0:
                                gain += 1
                        s = x + x
                        if TARGET_LO <= s <= TARGET_HI and counts_temp[s - TARGET_LO] == 0:
                            gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None or best_gain <= 0:
                        break
                    A_temp.append(best_x)
                    A_set.add(best_x)
                    A_temp.sort()
                    counts_temp = _pair_counts(A_temp)

                if _is_valid(counts_temp):
                    A_init = A_temp
                else:
                    continue  # Skip this restart if we couldn't fix it

            # Local Search
            A_opt, peak = _local_search(A_init, max_time=DEADLINE - 10)

            if peak < best_peak:
                best_peak = peak
                best_A = A_opt

            # Balance Polish
            if _elapsed() < DEADLINE - 5:
                A_polished, peak_polished = _balance_polish(A_opt, max_time=DEADLINE - 5)
                if peak_polished < best_peak:
                    best_peak = peak_polished
                    best_A = A_polished

            # Simulated Annealing (only if we have time)
            if _elapsed() < DEADLINE - 5 and best_peak > 3:
                A_sa, peak_sa = _simulated_annealing(
                    A_polished if _elapsed() < DEADLINE - 10 else A_opt,
                    max_time=DEADLINE - 5
                )
                if peak_sa < best_peak:
                    best_peak = peak_sa
                    best_A = A_sa

            # Early termination if peak is very low
            if best_peak <= 3:
                break

        if best_peak <= 3:
            break

    # Fallback
    if best_A is None:
        best_A = list(range(0, 129))

    return best_A
