import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    # Pre-compute neighbors (the ball of radius 1) for each point
    # neighbors[idx] is a list of 17 indices
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

    def greedy_set_cover(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []

        while not np.all(is_covered):
            uncovered_counts = np.sum(~is_covered[balls], axis=1)
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
            selected.append(best_idx)
            is_covered[balls[best_idx]] = True

        return selected

    def prune(selected_indices):
        if not selected_indices:
            return selected_indices

        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)

        for idx in selected_list:
            coverage_count[balls[idx]] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                if np.all(coverage_count[balls[idx]] > 1):
                    selected_list.pop(i)
                    coverage_count[balls[idx]] -= 1
                    changed = True
                    break

        return selected_list

    def local_search_optimize(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)

        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1

        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = random.Random(42)

        while time.time() < deadline:
            # Try to remove codewords
            removed = True
            while removed:
                removed = False
                for i in range(len(selected_list)):
                    if time.time() > deadline:
                        break
                    idx = selected_list[i]
                    if np.all(coverage_count[balls[idx]] > 1):
                        coverage_count[balls[idx]] -= 1
                        selected_list.pop(i)
                        removed = True
                        if len(selected_list) < best_len:
                            best_len = len(selected_list)
                            best_code = selected_list[:]
                        break

            if time.time() > deadline:
                break

            # Try swaps: move a codeword to a neighbor position
            n_sel = len(selected_list)
            if n_sel == 0:
                break

            # Pick a random codeword and try to move it
            progress = False
            for attempt in range(30):
                if time.time() > deadline:
                    break

                i = rng.randrange(n_sel)
                c = selected_list[i]

                # Try random neighbors of c
                nbrs = neighbors[c][:]
                rng.shuffle(nbrs)

                for c_prime in nbrs:
                    if c_prime == c:
                        continue
                    if c_prime in selected_list:
                        continue

                    # Tentatively swap
                    coverage_count[balls[c]] -= 1
                    coverage_count[balls[c_prime]] += 1

                    if np.any(coverage_count[balls[c]] == 0):
                        coverage_count[balls[c_prime]] -= 1
                        coverage_count[balls[c]] += 1
                        continue

                    selected_list[i] = c_prime
                    progress = True
                    break

                if progress:
                    break

            if progress:
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in pruned:
                        coverage_count[balls[idx]] += 1
                    selected_list = pruned

                if len(selected_list) < best_len:
                    best_len = len(selected_list)
                    best_code = selected_list[:]

        return best_code

    def simulated_annealing(selected_indices, deadline):
        """
        Simulated annealing over the set of codewords.
        Moves: remove a redundant codeword, add a new codeword, swap a codeword.
        Energy: number of uncovered points + lambda * number of codewords.
        """
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)

        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[balls[idx]] += 1

        uncovered = np.sum(coverage_count == 0)
        current_energy = uncovered + 0.01 * len(selected_list)
        best_len = len(selected_list)
        best_code = selected_list[:]

        rng = random.Random(123)
        start_temp = 10.0
        end_temp = 0.01
        cooling_rate = 0.999

        temp = start_temp
        iterations = 0

        while time.time() < deadline and temp > end_temp:
            iterations += 1
            n_sel = len(selected_list)

            # Choose a random move type
            move_type = rng.random()
            if move_type < 0.3 and n_sel > 0:
                # Remove a random codeword
                i = rng.randrange(n_sel)
                idx = selected_list[i]
                coverage_count[balls[idx]] -= 1
                new_uncovered = np.sum(coverage_count == 0)
                new_energy = new_uncovered + 0.01 * (n_sel - 1)
                delta = new_energy - current_energy

                if delta <= 0 or rng.random() < np.exp(-delta / temp):
                    selected_list.pop(i)
                    current_energy = new_energy
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                else:
                    coverage_count[balls[idx]] += 1

            elif move_type < 0.6:
                # Add a random codeword
                new_idx = rng.randrange(total_points)
                if new_idx not in selected_list:
                    coverage_count[balls[new_idx]] += 1
                    new_uncovered = np.sum(coverage_count == 0)
                    new_energy = new_uncovered + 0.01 * (n_sel + 1)
                    delta = new_energy - current_energy

                    if delta <= 0 or rng.random() < np.exp(-delta / temp):
                        selected_list.append(new_idx)
                        current_energy = new_energy
                    else:
                        coverage_count[balls[new_idx]] -= 1

            else:
                # Swap: remove one, add another
                if n_sel > 0:
                    i = rng.randrange(n_sel)
                    old_idx = selected_list[i]
                    new_idx = rng.randrange(total_points)
                    if new_idx not in selected_list:
                        coverage_count[balls[old_idx]] -= 1
                        coverage_count[balls[new_idx]] += 1
                        new_uncovered = np.sum(coverage_count == 0)
                        new_energy = new_uncovered + 0.01 * n_sel
                        delta = new_energy - current_energy

                        if delta <= 0 or rng.random() < np.exp(-delta / temp):
                            selected_list[i] = new_idx
                            current_energy = new_energy
                            if len(selected_list) < best_len:
                                best_len = len(selected_list)
                                best_code = selected_list[:]
                        else:
                            coverage_count[balls[new_idx]] -= 1
                            coverage_count[balls[old_idx]] += 1

            temp *= cooling_rate

            # Periodic pruning
            if iterations % 1000 == 0:
                pruned = prune(selected_list)
                if len(pruned) < len(selected_list):
                    coverage_count = np.zeros(total_points, dtype=np.int32)
                    for idx in pruned:
                        coverage_count[balls[idx]] += 1
                    selected_list = pruned
                    current_energy = np.sum(coverage_count == 0) + 0.01 * len(selected_list)
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]

        return best_code

    start_time = time.time()
    deadline = start_time + 110.0

    # Run greedy set cover with different seeds
    best_code = None
    best_len = float('inf')

    for seed in range(12):
        if time.time() > deadline - 40:
            break
        selected = greedy_set_cover(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected[:]

    if best_code is None:
        best_code = greedy_set_cover(0)
        best_code = prune(best_code)
        best_len = len(best_code)

    # Local search on the best code
    best_code = local_search_optimize(best_code, deadline)

    # Simulated annealing on the best code
    if time.time() < deadline - 5:
        sa_code = simulated_annealing(best_code, deadline)
        if len(sa_code) < len(best_code):
            best_code = sa_code

    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))

    return result
