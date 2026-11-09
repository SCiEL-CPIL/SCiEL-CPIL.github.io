import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128
    DOMAIN_SIZE = MAX_VAL + 1  # 129

    def calc_reps(A):
        """Compute representation counts for sums N_LOW..N_HIGH using unordered pairs.
        Uses convolution for speed."""
        if not A:
            return np.zeros(RANGE_SIZE, dtype=np.int32)

        # Faster indicator creation
        ind = np.bincount(list(A), minlength=DOMAIN_SIZE).astype(np.int32)

        # Convolution: ordered pairs count for each sum
        conv = np.convolve(ind, ind, mode='full')

        # We only need sums in [N_LOW, N_HIGH]
        conv_slice = conv[N_LOW:N_HIGH+1]

        # Vectorized diagonal correction
        s_vals = np.arange(N_LOW, N_HIGH + 1)
        half_s = s_vals // 2
        is_even = (s_vals % 2 == 0)
        diag = np.where(is_even, ind[half_s], 0)

        reps = (conv_slice + diag) // 2
        return reps

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    # --- Phase 1: Construct a structured initial candidate via greedy peeling ---
    # Start with full set [0, 128] and greedily remove elements to minimize peak
    best_A = None
    best_peak = float('inf')

    A_current = set(range(0, DOMAIN_SIZE))
    reps = calc_reps(A_current)

    if is_covered(reps):
        current_peak = get_peak(reps)
        best_peak = current_peak
        best_A = list(A_current)

        # Greedy peeling: repeatedly try to remove an element that reduces peak
        # or maintains validity with same peak
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            current_peak = get_peak(reps)

            # Try removing each element, prioritize those in "hot" sums
            # Sort candidates: elements appearing in more peak-sum representations first
            peak_indices = [i for i in range(RANGE_SIZE) if reps[i] == current_peak]
            if not peak_indices:
                break

            # Count how many times each element appears in peak-sum pairs
            elem_peak_count = {}
            for idx in peak_indices:
                target_s = N_LOW + idx
                for a in A_current:
                    b = target_s - a
                    if a <= b and b in A_current:
                        elem_peak_count[a] = elem_peak_count.get(a, 0) + 1
                        if a != b:
                            elem_peak_count[b] = elem_peak_count.get(b, 0) + 1

            # Sort elements by peak involvement (descending)
            sorted_elems = sorted(A_current, key=lambda x: -elem_peak_count.get(x, 0))

            for x in sorted_elems:
                test_A = A_current - {x}
                if len(test_A) == 0:
                    continue
                test_reps = calc_reps(test_A)
                if is_covered(test_reps):
                    test_peak = get_peak(test_reps)
                    if test_peak < current_peak:
                        A_current = test_A
                        reps = test_reps
                        current_peak = test_peak
                        improved = True
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = list(A_current)
                        break
                    elif test_peak == current_peak:
                        A_current = test_A
                        reps = test_reps
                        improved = True
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = list(A_current)
                        break

            if current_peak <= 1:
                break

    if best_A is None:
        # Fallback
        A_current = set(range(64, 129))
        reps = calc_reps(A_current)
        if is_covered(reps):
            best_peak = get_peak(reps)
            best_A = list(A_current)

    # --- Phase 2: Local search with simulated annealing ---
    if best_A is not None and time.time() < deadline - 10:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)

        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.01
        no_improve_count = 0
        max_no_improve = 500

        while time.time() < deadline - 5 and T > min_T and no_improve_count < max_no_improve:
            if len(current_A) == 0:
                break
            if current_peak == 1:
                break

            # Choose move type: swap (remove one, add one) or just remove
            move_type = random.random()
            new_A = set(current_A)

            if move_type < 0.7 and len(new_A) > 1:
                # Swap: remove a random element, add a random new one
                item_to_remove = random.choice(list(new_A))
                new_item = random.randint(0, MAX_VAL)
                if new_item == item_to_remove:
                    T *= cooling_rate
                    continue
                new_A.remove(item_to_remove)
                new_A.add(new_item)
            elif move_type < 0.9 and len(new_A) > 1:
                # Just remove
                item_to_remove = random.choice(list(new_A))
                new_A.remove(item_to_remove)
            else:
                # Just add
                new_item = random.randint(0, MAX_VAL)
                if new_item in new_A:
                    T *= cooling_rate
                    continue
                new_A.add(new_item)

            if len(new_A) == 0:
                T *= cooling_rate
                continue

            new_reps = calc_reps(new_A)

            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                delta = new_peak - current_peak

                if delta < 0:
                    current_A = new_A
                    current_reps = new_reps
                    current_peak = new_peak
                    no_improve_count = 0
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(current_A)
                else:
                    prob = np.exp(-delta / max(T, 0.001))
                    if random.random() < prob:
                        current_A = new_A
                        current_reps = new_reps
                        current_peak = new_peak
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            else:
                no_improve_count += 1

            T *= cooling_rate

        # Try removal after local search
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for x in list(current_A):
                    test_A = current_A - {x}
                    if len(test_A) == 0:
                        continue
                    test_reps = calc_reps(test_A)
                    if is_covered(test_reps):
                        test_peak = get_peak(test_reps)
                        if test_peak <= current_peak:
                            current_A = test_A
                            current_reps = test_reps
                            current_peak = test_peak
                            improved = True
                            if current_peak < best_peak:
                                best_peak = current_peak
                                best_A = list(current_A)
                            break

            if is_covered(current_reps):
                peak = get_peak(current_reps)
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(current_A)

    # --- Phase 3: Targeted peak-breaking ---
    if best_A is not None and time.time() < deadline - 3:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)

        polish_iter = 0
        while time.time() < deadline - 3 and polish_iter < 8000:
            polish_iter += 1
            if current_peak <= 1:
                break

            peak_indices = [i for i in range(RANGE_SIZE) if current_reps[i] == current_peak]
            if not peak_indices:
                break
            target_s = N_LOW + random.choice(peak_indices)

            valid_removals = []
            for a in current_A:
                b = target_s - a
                if b in current_A and a <= b:
                    valid_removals.append(a)

            if not valid_removals:
                continue

            a_to_remove = random.choice(valid_removals)
            best_new_A = None
            best_new_peak = current_peak
            best_new_reps = None

            for _ in range(8):
                new_item = random.randint(0, MAX_VAL)
                if new_item == a_to_remove:
                    continue
                new_A = set(current_A)
                new_A.remove(a_to_remove)
                new_A.add(new_item)
                if len(new_A) == 0:
                    continue
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak < best_new_peak:
                        best_new_peak = new_peak
                        best_new_A = new_A
                        best_new_reps = new_reps

            if best_new_A is not None and best_new_peak < current_peak:
                current_A = best_new_A
                current_reps = best_new_reps
                current_peak = best_new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = list(current_A)

    # --- Phase 4: Final greedy peeling on the best found ---
    if best_A is not None and time.time() < deadline - 2:
        current_A = set(best_A)
        current_reps = calc_reps(current_A)
        current_peak = get_peak(current_reps)

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_peak = get_peak(current_reps)
            if current_peak <= 1:
                break

            for x in sorted(current_A):
                test_A = current_A - {x}
                if len(test_A) == 0:
                    continue
                test_reps = calc_reps(test_A)
                if is_covered(test_reps):
                    test_peak = get_peak(test_reps)
                    if test_peak <= current_peak:
                        current_A = test_A
                        current_reps = test_reps
                        current_peak = test_peak
                        improved = True
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = list(current_A)
                        break

    if best_A is None:
        return list(range(64, 129))

    return best_A
