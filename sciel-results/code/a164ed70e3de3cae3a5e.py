import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Fallback for other parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each point
    neighbors = [None] * total_points
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q

        nbrs = [idx]
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]

            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            nbrs.append(idx + delta)

            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            nbrs.append(idx + delta)

        neighbors[idx] = nbrs

    # Build balls as a numpy array for vectorized operations
    balls = np.array(neighbors, dtype=np.int32)  # shape (6561, 17)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def get_counts(code):
        counts = np.zeros(total_points, dtype=np.int32)
        for c in code:
            counts[balls[c]] += 1
        return counts

    def reduce_code(code, counts):
        """
        Iteratively removes any codeword whose removal leaves all its neighbors covered by others.
        Returns new_code and new_counts.
        """
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()

        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                # If all neighbors of c have count > 1, we can remove c
                if np.all(current_counts[balls[c]] > 1):
                    current_counts[balls[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
        return current_code, current_counts

    def greedy_potential(seed=None):
        """Potential-based greedy: maintain potential array for fast candidate selection."""
        rng = np.random.default_rng(seed) if seed is not None else None

        potential = np.full(total_points, ball_size, dtype=np.int32)
        uncovered_mask = np.ones(total_points, dtype=bool)
        is_code = np.zeros(total_points, dtype=bool)
        code_list = []

        while np.any(uncovered_mask):
            # Find word with highest potential (covers most uncovered points)
            # Use argmax, but handle ties randomly if rng provided
            max_pot = np.max(potential)
            if max_pot == 0:
                break

            candidates = np.where(potential == max_pot)[0]
            if len(candidates) > 1 and rng is not None:
                c = int(rng.choice(candidates))
            else:
                c = int(candidates[0])

            code_list.append(c)
            is_code[c] = True
            potential[c] = 0  # Prevent re-selection

            # Update potential for all words that cover any newly-covered point
            for x in balls[c]:
                if uncovered_mask[x]:
                    uncovered_mask[x] = False
                    # All words that cover x lose potential
                    potential[balls[x]] -= 1
                    # Ensure non-negative
                    np.maximum(potential, 0, out=potential)

        return code_list

    def local_search_walk(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list, coverage_count = reduce_code(selected_list, get_counts(selected_list))
        selected_set = set(selected_list)

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        while time.time() < deadline:
            # 1. Remove any removable codeword (full sweep, deterministic)
            removed_any = True
            while removed_any and time.time() < deadline:
                removed_any = False
                for i in range(len(selected_list)):
                    idx = selected_list[i]
                    if np.all(coverage_count[balls[idx]] > 1):
                        coverage_count[balls[idx]] -= 1
                        selected_list.pop(i)
                        selected_set.discard(idx)
                        removed_any = True
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        break

            if time.time() > deadline:
                break

            # 2. Neighbor-restricted swap: move a random codeword to a point in its own ball
            n_sel = len(selected_list)
            if n_sel == 0:
                break

            progress = False
            # Try a few swaps before giving up on this iteration
            for _ in range(30):
                if time.time() > deadline:
                    break
                i = rng.randrange(n_sel)
                c = selected_list[i]
                nbrs = list(neighbors[c])
                rng.shuffle(nbrs)

                for c_prime in nbrs:
                    if c_prime == c or c_prime in selected_set:
                        continue

                    # Tentatively swap: remove c, add c_prime
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1

                    # Check validity: no point in balls[c] should have count 0
                    if np.any(coverage_count[balls[c]] == 0):
                        # Invalid, revert
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        continue

                    # Valid swap
                    selected_list[i] = c_prime
                    selected_set.discard(c)
                    selected_set.add(c_prime)
                    progress = True
                    break

                if progress:
                    break

            if progress:
                # After a swap, prune to catch newly-redundant codewords
                pruned, new_counts = reduce_code(selected_list, coverage_count)
                if len(pruned) < len(selected_list):
                    coverage_count = new_counts
                    selected_list = pruned
                    selected_set = set(selected_list)

                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    def polish(selected_indices, deadline):
        """Final high-precision pass: full deterministic sweeps + exhaustive neighbor checks for removals."""
        selected_list = list(selected_indices)
        coverage_count = get_counts(selected_list)
        selected_set = set(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]

        # 1. Aggressive deterministic removal
        for _ in range(100):
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    coverage_count[balls[idx]] -= 1
                    selected_list.pop(i)
                    selected_set.discard(idx)
                    changed = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
            if not changed:
                break

        # 2. Try single-step moves that might enable removal
        rng = random.Random(99)
        attempts = 0
        max_attempts = 5000  # Limit to prevent hanging

        while time.time() < deadline and attempts < max_attempts:
            if not selected_list:
                break

            # Pick a random codeword
            i = rng.randrange(len(selected_list))
            c = selected_list[i]
            nbrs = list(neighbors[c])
            rng.shuffle(nbrs)

            for c_prime in nbrs:
                if c_prime == c or c_prime in selected_set:
                    continue

                # Tentative move
                coverage_count[balls[c]] -= 1
                coverage_count[balls[c_prime]] += 1

                if np.any(coverage_count[balls[c]] == 0):
                    coverage_count[balls[c_prime]] -= 1
                    coverage_count[balls[c]] += 1
                    attempts += 1
                    break

                # Move is valid. Now check if we can remove any codeword.
                removed_any = False
                removed_idx = -1
                for j in range(len(selected_list)):
                    idx_j = selected_list[j]
                    if np.all(coverage_count[balls[idx_j]] > 1):
                        removed_any = True
                        removed_idx = j
                        break

                if removed_any:
                    # Accept the move and the removal
                    val = selected_list[removed_idx]
                    coverage_count[balls[val]] -= 1
                    selected_list.pop(removed_idx)
                    selected_set.discard(val)

                    # Update best
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]

                    attempts += 1
                    break  # Restart the outer loop with the new state
                else:
                    # No removal possible, revert the move
                    coverage_count[balls[c_prime]] -= 1
                    coverage_count[balls[c]] += 1
                    attempts += 1
                    break

        return best_code

    def is_valid_code(selected_list):
        """Check if the selected codewords cover all points."""
        if not selected_list:
            return total_points == 0
        coverage_count = get_counts(selected_list)
        return bool(np.all(coverage_count > 0))

    # 1. Generate initial codes using potential-based greedy
    best_code = None
    best_len = float('inf')

    # Deterministic greedy
    selected = greedy_potential(seed=None)
    selected, _ = reduce_code(selected, get_counts(selected))
    if len(selected) < best_len:
        best_len = len(selected)
        best_code = selected[:]

    # Random greedy with multiple seeds
    for seed in range(8):
        if time.time() > deadline - 40:
            break
        selected = greedy_potential(seed=seed)
        selected, _ = reduce_code(selected, get_counts(selected))
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

    if best_code is None:
        best_code = greedy_potential(seed=None)
        best_code, _ = reduce_code(best_code, get_counts(best_code))
        best_len = len(best_code)

    # 2. Local Search on the best initial code
    best_code = local_search_walk(best_code, deadline - 15)

    # 3. Polish
    best_code = polish(best_code, deadline)

    # Verify validity before returning
    if not is_valid_code(best_code):
        # Fallback to a known valid construction
        best_code = greedy_potential(seed=None)
        best_code, _ = reduce_code(best_code, get_counts(best_code))

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
