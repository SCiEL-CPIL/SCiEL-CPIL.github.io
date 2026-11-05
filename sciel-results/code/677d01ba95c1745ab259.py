"""Additive bases with few representations - combined approach."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    n = 128
    min_sum = 128
    max_sum = 256

    def compute_counts_fast(A_set):
        """Fast computation of pair-sum counts using numpy convolution."""
        indicator = np.zeros(n + 1, dtype=np.int64)
        for x in A_set:
            indicator[x] = 1
        conv = np.convolve(indicator, indicator, mode='full')
        # conv[s] = ordered pair count for sum s
        # Unordered: for s even, if s/2 in A, diag=1 else 0
        # unordered = (conv[s] + diag) // 2
        counts = np.zeros(max_sum + 1, dtype=np.int64)
        for s in range(min_sum, max_sum + 1):
            if s < len(conv):
                ordered = conv[s]
                if s % 2 == 0:
                    half = s // 2
                    if 0 <= half <= n and indicator[half] == 1:
                        diag = 1
                    else:
                        diag = 0
                else:
                    diag = 0
                unordered = (ordered + diag) // 2
                counts[s] = unordered
        return counts

    def get_cost(A_list):
        A_set = set(A_list)
        counts = compute_counts_fast(A_set)
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return float('inf')
        return int(np.max(target_counts))

    def get_cost_with_counts(A_list):
        A_set = set(A_list)
        counts = compute_counts_fast(A_set)
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return float('inf'), counts
        return int(np.max(target_counts)), counts

    best_score = float('inf')
    best_set = None

    def try_set(A_list):
        nonlocal best_score, best_set
        A_sorted = sorted(A_list)
        c = get_cost(A_sorted)
        if c < best_score:
            best_score = c
            best_set = A_sorted[:]

    # Initial solution: the seed
    initial = list(range(64, 129))
    try_set(initial)

    # Try structured initial solutions
    # Symmetric sets around 64
    for offset in range(0, 30):
        lo = max(0, 64 - offset)
        hi = min(128, 64 + offset)
        candidate = list(range(lo, hi + 1))
        if len(candidate) > 0:
            try_set(candidate)

    # Modular filter approach: x mod 3 != 2 gives residues {0,1}
    A_mod3 = [x for x in range(n + 1) if x % 3 != 2]
    try_set(A_mod3)

    # Try other modular patterns
    for mod in [4, 5, 6, 7, 8]:
        for keep in range(1, mod):
            A_mod = [x for x in range(n + 1) if x % mod != mod - keep]
            if len(A_mod) > 0:
                try_set(A_mod)

    # Random initialization with various sizes
    random.seed(42)
    for size in range(30, 80):
        if time.time() > deadline - 15:
            break
        for _ in range(15):
            if time.time() > deadline - 12:
                break
            A = random.sample(range(0, 129), size)
            try_set(A)

    # Local search: swap-based improvement with fast cost
    if best_set is not None and time.time() < deadline - 10:
        current = best_set[:]
        current_cost = best_score
        in_set = set(current)

        while time.time() < deadline - 5:
            improved = False

            # Try removing elements
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                val = current[i]
                trial = current[:i] + current[i+1:]
                if len(trial) == 0:
                    continue
                tc = get_cost(trial)
                if tc <= current_cost:
                    current = trial
                    current_cost = tc
                    in_set.discard(val)
                    improved = True
                    break

            # Try adding elements
            if not improved:
                for new_val in range(129):
                    if time.time() > deadline - 5:
                        break
                    if new_val in in_set:
                        continue
                    trial = current + [new_val]
                    tc = get_cost(trial)
                    if tc < current_cost:
                        current = trial
                        current_cost = tc
                        in_set.add(new_val)
                        improved = True
                        break

            # Try swaps - only try a subset for speed
            if not improved:
                # Try swapping each element with a few candidates
                for i in range(len(current)):
                    if time.time() > deadline - 5:
                        break
                    old_val = current[i]
                    # Try nearby values first, then random
                    candidates = []
                    for delta in range(-5, 6):
                        nv = old_val + delta
                        if 0 <= nv <= 128 and nv != old_val and nv not in in_set:
                            candidates.append(nv)
                    # Add some random candidates
                    for _ in range(5):
                        nv = random.randint(0, 128)
                        if nv != old_val and nv not in in_set:
                            candidates.append(nv)

                    for new_val in candidates:
                        if time.time() > deadline - 5:
                            break
                        trial = current[:]
                        trial[i] = new_val
                        trial.sort()
                        tc = get_cost(trial)
                        if tc < current_cost:
                            current = trial
                            current_cost = tc
                            in_set.discard(old_val)
                            in_set.add(new_val)
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                break

        if current_cost < best_score:
            best_score = current_cost
            best_set = current[:]

    # Second phase: more aggressive search if we have time and score > 3
    if time.time() < deadline - 8 and best_score > 3:
        random.seed(123)
        for _ in range(50):
            if time.time() > deadline - 6:
                break
            # Start from best and perturb
            if best_set is None:
                break
            size = len(best_set) + random.randint(-5, 5)
            size = max(20, min(80, size))
            A = random.sample(range(0, 129), size)
            try_set(A)
            if best_score <= 3:
                break

    # Final local search
    if best_set is not None and time.time() < deadline - 3:
        current = best_set[:]
        current_cost = best_score
        in_set = set(current)

        while time.time() < deadline - 1:
            improved = False

            # Try removing
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline - 1:
                    break
                val = current[i]
                trial = current[:i] + current[i+1:]
                if len(trial) == 0:
                    continue
                tc = get_cost(trial)
                if tc <= current_cost:
                    current = trial
                    current_cost = tc
                    in_set.discard(val)
                    improved = True
                    break

            # Try adding
            if not improved:
                for new_val in range(129):
                    if time.time() > deadline - 1:
                        break
                    if new_val in in_set:
                        continue
                    trial = current + [new_val]
                    tc = get_cost(trial)
                    if tc < current_cost:
                        current = trial
                        current_cost = tc
                        in_set.add(new_val)
                        improved = True
                        break

            # Try swaps - limited candidates
            if not improved:
                for i in range(len(current)):
                    if time.time() > deadline - 1:
                        break
                    old_val = current[i]
                    candidates = []
                    for delta in range(-3, 4):
                        nv = old_val + delta
                        if 0 <= nv <= 128 and nv != old_val and nv not in in_set:
                            candidates.append(nv)

                    for new_val in candidates:
                        if time.time() > deadline - 1:
                            break
                        trial = current[:]
                        trial[i] = new_val
                        trial.sort()
                        tc = get_cost(trial)
                        if tc < current_cost:
                            current = trial
                            current_cost = tc
                            in_set.discard(old_val)
                            in_set.add(new_val)
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                break

        if current_cost < best_score:
            best_score = current_cost
            best_set = current[:]

    if best_set is None or len(best_set) == 0:
        return list(range(64, 129))

    return list(best_set)

# EVOLVE-BLOCK-END
