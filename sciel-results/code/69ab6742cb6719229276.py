"""Max-SAT / WalkSAT-style local search for additive bases with few representations."""
import time
import random
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    n = 128
    target_lo = 128
    target_hi = 256
    target_range = target_hi - target_lo + 1  # 129 values

    # We represent A as a boolean numpy array of length n+1
    # We maintain counts[s] = number of unordered pairs (a,b), a<=b, a+b=s, a,b in A
    # for s in [0, 256]

    def make_A_from_set(A_set):
        arr = np.zeros(n + 1, dtype=bool)
        for x in A_set:
            arr[x] = True
        return arr

    def compute_counts(A_arr):
        """Compute unordered pair counts for all sums 0..256."""
        counts = np.zeros(257, dtype=np.int64)
        indices = np.where(A_arr)[0]
        for i in range(len(indices)):
            a = indices[i]
            for j in range(i, len(indices)):
                b = indices[j]
                counts[a + b] += 1
        return counts

    def get_peak_and_invalid(counts):
        target_counts = counts[target_lo:target_hi + 1]
        peak = int(np.max(target_counts))
        invalid_count = int(np.sum(target_counts == 0))
        return peak, invalid_count

    def cost(peak, invalid):
        """Cost function: heavily penalize uncovered targets, then minimize peak."""
        return peak * 1000 + invalid

    # Initial solution: A = [64, 128] (the seed)
    A = make_A_from_set(set(range(64, 129)))
    counts = compute_counts(A)
    peak, invalid = get_peak_and_invalid(counts)
    current_cost = cost(peak, invalid)

    best_A = A.copy()
    best_counts = counts.copy()
    best_peak, best_invalid = peak, invalid
    best_cost = current_cost

    # WalkSAT-style local search
    # We flip a single bit (add or remove an element) at each step.
    # We choose which bit to flip based on a probabilistic heuristic:
    #   - With probability p_random, pick a random bit to flip.
    #   - Otherwise, pick the bit whose flip minimizes the cost (greedy).
    #   - If the flip increases cost, accept with probability exp(-delta/T) (simulated annealing).

    random.seed(42)

    T = 20.0
    T_min = 0.5
    cooling = 0.9995
    p_random = 0.3

    # Precompute for fast incremental updates:
    # When we toggle bit x:
    #   If adding x: for each y in A with y != x, add 1 to counts[x+y]; add 1 to counts[2x]
    #   If removing x: for each y in A with y != x, subtract 1 from counts[x+y]; subtract 1 from counts[2x]

    def toggle_bit(A_arr, counts_arr, x):
        """Toggle bit x in A_arr and update counts_arr incrementally. Returns the number of elements in A after toggle."""
        if A_arr[x]:
            # Removing x
            for y in range(n + 1):
                if A_arr[y] and y != x:
                    counts_arr[x + y] -= 1
            counts_arr[2 * x] -= 1
            A_arr[x] = False
        else:
            # Adding x
            for y in range(n + 1):
                if A_arr[y] and y != x:
                    counts_arr[x + y] += 1
            counts_arr[2 * x] += 1
            A_arr[x] = True
        return int(np.sum(A_arr))

    iteration = 0
    stagnation_counter = 0
    max_stagnation = 5000

    while time.time() < deadline and T > T_min:
        iteration += 1

        # Choose which bit to flip
        if random.random() < p_random:
            x = random.randint(0, n)
        else:
            # Greedy: find the bit whose flip gives the best cost improvement
            best_x = -1
            best_delta = float('inf')
            # Sample a subset of bits for speed (not all 129)
            candidates = list(range(n + 1))
            random.shuffle(candidates)
            for x in candidates[:40]:  # evaluate up to 40 candidates
                # Simulate toggle
                A_temp = A.copy()
                counts_temp = counts.copy()
                toggle_bit(A_temp, counts_temp, x)
                p, inv = get_peak_and_invalid(counts_temp)
                c = cost(p, inv)
                delta = c - current_cost
                if delta < best_delta:
                    best_delta = delta
                    best_x = x
            if best_x >= 0:
                x = best_x
            else:
                x = random.randint(0, n)

        # Perform the toggle
        A_copy = A.copy()
        counts_copy = counts.copy()
        toggle_bit(A, counts, x)

        peak_new, invalid_new = get_peak_and_invalid(counts)
        new_cost = cost(peak_new, invalid_new)
        delta = new_cost - current_cost

        # Acceptance criterion
        if delta <= 0 or (T > T_min and random.random() < np.exp(-delta / T)):
            current_cost = new_cost
            peak = peak_new
            invalid = invalid_new

            if new_cost < best_cost:
                best_cost = new_cost
                best_A = A.copy()
                best_counts = counts.copy()
                best_peak = peak_new
                best_invalid = invalid_new
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Adaptive p_random: increase randomness when stagnating
            if stagnation_counter > max_stagnation:
                p_random = min(0.7, p_random + 0.05)
                stagnation_counter = 0
        else:
            # Revert
            A[:] = A_copy
            counts[:] = counts_copy

        T *= cooling

        # Periodic restart: if stuck for too long, reset to best and reheat
        if iteration % 20000 == 0 and time.time() < deadline:
            if best_cost < current_cost:
                A[:] = best_A
                counts[:] = best_counts
                current_cost = best_cost
                peak, invalid = best_peak, best_invalid
                T = max(T, 5.0)  # reheat

    # Final verification and return
    # Ensure best solution is valid (all targets covered)
    if best_invalid > 0:
        # Fallback: use the seed solution
        best_A = make_A_from_set(set(range(64, 129)))

    A_list = np.where(best_A)[0].tolist()
    if len(A_list) == 0:
        A_list = list(range(64, 129))

    return sorted(A_list)
