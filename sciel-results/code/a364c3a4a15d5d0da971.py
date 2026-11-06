import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s kill limit

    N = 1000
    MAX_SUM = 2 * N  # max possible sum is 2000

    def build_counts(chosen):
        """Build sum-count dict from a list of chosen elements."""
        counts = defaultdict(int)
        for i, x in enumerate(chosen):
            counts[2 * x] += 1
            for j in range(i + 1, len(chosen)):
                counts[x + chosen[j]] += 1
        return counts

    def can_add_fast(x, chosen_set, counts_arr):
        """Check if x can be added using numpy array for counts."""
        if x in chosen_set:
            return False
        if counts_arr[2 * x] >= 2:
            return False
        # Check all x + y for y in chosen_set
        # We need to check if any count[x+y] >= 2
        # This is hard to vectorize without knowing chosen elements
        return True  # placeholder, will use slower check below

    def can_add_slow(x, chosen_set, counts):
        """Slow but correct check."""
        if x in chosen_set:
            return False
        s2 = 2 * x
        if counts.get(s2, 0) >= 2:
            return False
        for y in chosen_set:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        return True

    def greedy_build(seed=42, noise_scale=0.0):
        """Fast greedy using numpy for scoring."""
        random.seed(seed)
        np.random.seed(seed)
        chosen = []
        chosen_set = set()
        counts_arr = np.zeros(MAX_SUM + 1, dtype=np.int32)
        available = np.ones(N + 1, dtype=bool)
        available[0] = False

        while True:
            if time.time() > deadline:
                break

            if len(chosen) == 0:
                # First element: any works, pick smallest for determinism or random
                if noise_scale > 0:
                    best_x = random.randint(1, N)
                else:
                    best_x = 1
                chosen.append(best_x)
                chosen_set.add(best_x)
                available[best_x] = False
                counts_arr[2 * best_x] += 1
                continue

            # Get available candidates
            avail_indices = np.where(available[1:])[0] + 1
            if len(avail_indices) == 0:
                break

            # Check 2*x validity
            doubles = 2 * avail_indices
            double_counts = counts_arr[doubles]
            invalid_double = double_counts >= 2
            tight_double = (double_counts == 1).astype(np.int32)

            invalid_mask = invalid_double.copy()
            tight_mask = tight_double.copy()

            # Iterate over chosen elements, but break early if all invalid
            for y in chosen:
                sums = avail_indices + y
                # Only need sums in valid range [1, MAX_SUM]
                mask = (sums >= 1) & (sums <= MAX_SUM)
                if not np.all(mask):
                    sums = sums[mask]
                    invalid_mask = invalid_mask[mask]
                    tight_mask = tight_mask[mask]
                
                if len(sums) == 0:
                    break
                    
                sum_counts = counts_arr[sums]
                new_invalid = sum_counts >= 2
                new_tight = (sum_counts == 1).astype(np.int32)
                
                invalid_mask = invalid_mask | new_invalid
                tight_mask = tight_mask + new_tight
                
                # Early exit if no valid candidates remain
                if not np.any(~invalid_mask):
                    break

            # Filter out invalid candidates
            valid_mask = ~invalid_mask
            if not np.any(valid_mask):
                break

            valid_indices = avail_indices[valid_mask]
            valid_tight = tight_mask[valid_mask]

            # Add noise if needed
            if noise_scale > 0:
                noise = np.random.random(len(valid_indices)) * noise_scale
                scores = valid_tight.astype(np.float64) + noise
            else:
                scores = valid_tight.astype(np.float64)

            best_idx = np.argmin(scores)
            best_x = int(valid_indices[best_idx])

            # Update structures
            chosen.append(best_x)
            chosen_set.add(best_x)
            available[best_x] = False

            # Update counts
            counts_arr[2 * best_x] += 1
            for y in chosen[:-1]:
                counts_arr[best_x + y] += 1

        return chosen, build_counts(chosen), chosen_set

    def local_search(chosen, counts, chosen_set, max_iters=500):
        """Improve set by trying additions and swaps."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        counts_arr = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for s, c in counts.items():
            counts_arr[s] = c
        chosen_arr = np.array(chosen, dtype=np.int32)

        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1

            # Try to add any missing element (vectorized check)
            all_candidates = np.arange(1, N + 1)
            in_set = np.zeros(N + 1, dtype=bool)
            in_set[chosen_arr] = True
            candidates = all_candidates[~in_set[all_candidates]]
            
            if len(candidates) > 0:
                # Check 2*x
                doubles = 2 * candidates
                double_ok = counts_arr[doubles] < 2
                candidates = candidates[double_ok]
                
                if len(candidates) > 0:
                    # Check x+y for all y in chosen (vectorized)
                    valid_mask = np.ones(len(candidates), dtype=bool)
                    for y in chosen_arr:
                        sums = candidates + y
                        if np.any(counts_arr[sums] >= 2):
                            valid_mask &= counts_arr[sums] < 2
                        if not np.any(valid_mask):
                            break
                    if np.any(valid_mask):
                        best_x = int(candidates[np.argmax(valid_mask)])
                        chosen.append(best_x)
                        chosen_set.add(best_x)
                        chosen_arr = np.append(chosen_arr, best_x)
                        counts_arr[2 * best_x] += 1
                        for y in chosen_arr[:-1]:
                            counts_arr[best_x + y] += 1
                        improved = True

            if improved:
                continue

            # No simple addition; try swaps (optimized)
            # For each rem, find best y to swap in
            # Use a faster approach: precompute which y are valid for each rem
            found_swap = False
            for rem in chosen[:]:
                if time.time() > deadline:
                    break

                # Temporarily remove rem
                temp_counts = counts_arr.copy()
                temp_counts[2 * rem] -= 1
                for z in chosen:
                    if z != rem:
                        temp_counts[rem + z] -= 1

                # Find best y to add (vectorized)
                all_candidates = np.arange(1, N + 1)
                in_set = np.zeros(N + 1, dtype=bool)
                in_set[chosen_arr] = True
                in_set[rem] = False  # rem is removed
                candidates = all_candidates[~in_set[all_candidates]]
                
                if len(candidates) == 0:
                    continue

                # Check 2*y
                doubles = 2 * candidates
                double_ok = temp_counts[doubles] < 2
                candidates = candidates[double_ok]
                
                if len(candidates) == 0:
                    continue

                # Check y+z for all z in chosen (excluding rem)
                chosen_minus_rem = [z for z in chosen if z != rem]
                valid_mask = np.ones(len(candidates), dtype=bool)
                tight_arr = np.zeros(len(candidates), dtype=np.int32)
                for z in chosen_minus_rem:
                    sums = candidates + z
                    sc = temp_counts[sums]
                    valid_mask &= sc < 2
                    tight_arr += (sc == 1).astype(np.int32)
                    if not np.any(valid_mask):
                        break

                if not np.any(valid_mask):
                    continue

                valid_candidates = candidates[valid_mask]
                valid_tight = tight_arr[valid_mask]
                best_idx = np.argmin(valid_tight)
                best_y = int(valid_candidates[best_idx])

                # Perform swap
                counts_arr[2 * rem] -= 1
                for z in chosen:
                    if z != rem:
                        counts_arr[rem + z] -= 1

                chosen.remove(rem)
                chosen_set.remove(rem)
                chosen_arr = np.array(chosen, dtype=np.int32)

                chosen.append(best_y)
                chosen_set.add(best_y)
                chosen_arr = np.append(chosen_arr, best_y)
                counts_arr[2 * best_y] += 1
                for z in chosen_arr[:-1]:
                    counts_arr[best_y + z] += 1

                improved = True
                found_swap = True
                break

        return chosen, build_counts(chosen), chosen_set

    def polish(chosen, counts, chosen_set, max_rounds=50):
        """Final polish: repeatedly try to add any element."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        counts_arr = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for s, c in counts.items():
            counts_arr[s] = c

        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            added = False
            for x in range(1, N + 1):
                if x in chosen_set:
                    continue
                if counts_arr[2 * x] >= 2:
                    continue
                valid = True
                for y in chosen_set:
                    if counts_arr[x + y] >= 2:
                        valid = False
                        break
                if valid:
                    chosen.append(x)
                    chosen_set.add(x)
                    counts_arr[2 * x] += 1
                    for y in chosen:
                        if y != x:
                            counts_arr[x + y] += 1
                    added = True
                    break
            if not added:
                break

        return chosen, build_counts(chosen), chosen_set

    best_set = []
    best_size = 0

    # Phase 1: Deterministic greedy with various seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        chosen, counts, chosen_set = greedy_build(seed=seed, noise_scale=0.0)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen

    # Phase 2: Local search on best found
    if best_set and time.time() < deadline:
        counts = build_counts(best_set)
        chosen_set = set(best_set)
        best_set, counts, chosen_set = local_search(best_set, counts, chosen_set, max_iters=500)
        best_size = len(best_set)

    # Phase 3: Randomized greedy restarts with local search
    while time.time() < deadline:
        seed = random.randint(0, 100000)
        noise = random.choice([0.0, 0.1, 0.3, 0.5, 1.0])
        chosen, counts, chosen_set = greedy_build(seed=seed, noise_scale=noise)

        # Run local search if it looks promising
        if len(chosen) >= best_size - 5 and time.time() < deadline:
            counts = build_counts(chosen)
            chosen_set = set(chosen)
            chosen, counts, chosen_set = local_search(chosen, counts, chosen_set, max_iters=250)

        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen

    # Phase 4: Final polish
    if best_set and time.time() < deadline:
        counts = build_counts(best_set)
        chosen_set = set(best_set)
        best_set, counts, chosen_set = polish(best_set, counts, chosen_set, max_rounds=200)
        best_size = len(best_set)

    return best_set
