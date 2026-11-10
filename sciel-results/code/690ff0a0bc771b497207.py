import time
import random
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    start_time = time.time()
    timeout = 115  # Safety margin for 150s wall-clock limit

    def build_counts(chosen):
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        arr = np.array(chosen, dtype=np.int32)
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                counts[arr[i] + arr[j]] += 1
        return counts

    def can_add_fast(chosen_arr, counts, x):
        """Vectorized check if x can be added."""
        if len(chosen_arr) == 0:
            return counts[2 * x] < 2
        # Check all sums x + y for y in chosen
        sums = chosen_arr + x
        # Check if any of these sums already have count 2
        # Also check 2*x
        if counts[2 * x] >= 2:
            return False
        # Check sums involving x
        # We need counts[s] + 1 <= 2 for all s in sums, i.e., counts[s] < 2
        # But we also need to handle the case where x is already in chosen (shouldn't happen)
        return np.all(counts[sums] < 2)

    def add_fast(chosen_arr, counts, x, in_set):
        sums = chosen_arr + x
        counts[sums] += 1
        counts[2 * x] += 1
        in_set[x] = True
        return np.append(chosen_arr, x)

    def remove_fast(chosen_arr, counts, x, in_set):
        # Remove x from sums
        others = chosen_arr[chosen_arr != x]
        counts[others + x] -= 1
        counts[2 * x] -= 1
        in_set[x] = False
        return chosen_arr[chosen_arr != x]

    def get_congestion(counts):
        return int(np.sum(counts * counts))

    def greedy_random(rng, order='random'):
        if order == 'random':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        elif order == 'descending':
            candidates = list(range(N, 0, -1))
        elif order == 'sparse_first':
            odds = list(range(1, N + 1, 2))
            evens = list(range(2, N + 1, 2))
            rng.shuffle(odds)
            rng.shuffle(evens)
            candidates = odds + evens
        elif order == 'prime_first':
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [i for i in range(1, N + 1) if is_prime(i)]
            non_primes = [i for i in range(1, N + 1) if not is_prime(i)]
            rng.shuffle(primes)
            rng.shuffle(non_primes)
            candidates = primes + non_primes
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)

        chosen_arr = np.array([], dtype=np.int32)
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        in_set = np.zeros(N + 1, dtype=bool)

        for x in candidates:
            if can_add_fast(chosen_arr, counts, x):
                chosen_arr = add_fast(chosen_arr, counts, x, in_set)

        return chosen_arr, counts, in_set

    def local_search_congestion(chosen_arr, counts, in_set, max_rounds=80, start_time=None, timeout=None):
        """Local search using congestion reduction to guide swaps."""
        improved = True
        rounds = 0
        current_congestion = get_congestion(counts)

        while improved and rounds < max_rounds:
            improved = False
            rounds += 1

            if timeout and start_time:
                if time.time() - start_time > timeout:
                    break

            # Phase 1: Try to add any element
            for x in range(1, N + 1):
                if not in_set[x] and can_add_fast(chosen_arr, counts, x):
                    chosen_arr = add_fast(chosen_arr, counts, x, in_set)
                    improved = True
                    current_congestion = get_congestion(counts)
                    break

            if improved:
                continue

            # Phase 2: Try congestion-reducing swaps
            # Find elements whose removal would reduce congestion the most
            # and elements whose addition would increase congestion the least

            # Sample a subset of elements to try removing for speed
            n_chosen = len(chosen_arr)
            if n_chosen == 0:
                break

            sample_size = min(n_chosen, 40)
            if sample_size < n_chosen:
                indices = np.random.choice(n_chosen, sample_size, replace=False)
                sampled = chosen_arr[indices]
            else:
                sampled = chosen_arr

            best_swap = None
            best_swap_congestion = current_congestion

            for x in sampled:
                # Temporarily remove x
                new_chosen = chosen_arr[chosen_arr != x]
                others = new_chosen
                temp_counts = counts.copy()
                temp_counts[others + x] -= 1
                temp_counts[2 * x] -= 1

                # Try to add elements that reduce congestion
                # Focus on elements that would add to sums with count 0 or 1
                # Sample candidates
                not_in_set = np.where(~in_set)[0]
                not_in_set = not_in_set[not_in_set >= 1]

                if len(not_in_set) == 0:
                    continue

                # Sample up to 50 candidates
                cand_sample_size = min(len(not_in_set), 50)
                if cand_sample_size < len(not_in_set):
                    cand_indices = np.random.choice(len(not_in_set), cand_sample_size, replace=False)
                    candidates = not_in_set[cand_indices]
                else:
                    candidates = not_in_set

                for y in candidates:
                    if y == x:
                        continue
                    # Check if y can be added to new_chosen
                    if len(new_chosen) > 0:
                        sums_y = new_chosen + y
                        if np.any(temp_counts[sums_y] >= 2):
                            continue
                    if temp_counts[2 * y] >= 2:
                        continue

                    # Calculate new congestion
                    new_counts = temp_counts.copy()
                    new_counts[new_chosen + y] += 1
                    new_counts[2 * y] += 1
                    new_congestion = int(np.sum(new_counts * new_counts))

                    if new_congestion < best_swap_congestion:
                        best_swap_congestion = new_congestion
                        best_swap = (x, y, new_chosen, new_counts)

            if best_swap is not None:
                x, y, new_chosen, new_counts = best_swap
                chosen_arr = new_chosen
                counts = new_counts
                in_set[x] = False
                in_set[y] = True
                improved = True
                current_congestion = best_swap_congestion

        return chosen_arr, counts, in_set

    def local_search_sa(chosen_arr, counts, in_set, start_time=None, timeout=None):
        """Simulated annealing-style local search."""
        if timeout and start_time:
            if time.time() - start_time > timeout:
                return chosen_arr, counts, in_set

        current_len = len(chosen_arr)
        best_chosen = chosen_arr.copy()
        best_counts = counts.copy()
        best_in_set = in_set.copy()

        temperature = 30.0
        cooling = 0.97

        while time.time() - start_time < timeout:
            if temperature < 0.5:
                break

            if len(chosen_arr) == 0:
                break

            x = int(random.choice(chosen_arr))
            y = random.randint(1, N)

            if in_set[y] or y == x:
                continue

            # Remove x
            others = chosen_arr[chosen_arr != x]
            counts[others + x] -= 1
            counts[2 * x] -= 1
            in_set[x] = False
            new_chosen = others

            # Try to add y
            valid = True
            if len(new_chosen) > 0:
                sums_y = new_chosen + y
                if np.any(counts[sums_y] >= 2):
                    valid = False
            if counts[2 * y] >= 2:
                valid = False

            if valid:
                counts[new_chosen + y] += 1
                counts[2 * y] += 1
                in_set[y] = True
                chosen_arr = np.append(new_chosen, y)
                new_len = len(chosen_arr)

                if new_len > current_len:
                    current_len = new_len
                    if new_len > len(best_chosen):
                        best_chosen = chosen_arr.copy()
                        best_counts = counts.copy()
                        best_in_set = in_set.copy()
                else:
                    delta = current_len - new_len
                    if delta > 0 and random.random() < np.exp(-delta / temperature):
                        current_len = new_len
                    else:
                        # Revert: remove y, put x back
                        counts[chosen_arr[chosen_arr != y] + y] -= 1
                        counts[2 * y] -= 1
                        in_set[y] = False
                        chosen_arr = chosen_arr[chosen_arr != y]

                        counts[chosen_arr + x] += 1
                        counts[2 * x] += 1
                        in_set[x] = True
                        chosen_arr = np.append(chosen_arr, x)
            else:
                # Put x back
                counts[others + x] += 1
                counts[2 * x] += 1
                in_set[x] = True
                chosen_arr = np.append(others, x)

            temperature *= cooling

        # Restore best
        chosen_arr = best_chosen
        counts = best_counts
        in_set = best_in_set

        return chosen_arr, counts, in_set

    best_set = np.array([], dtype=np.int32)
    best_len = 0

    # Phase 1: Quick greedy with diverse orderings
    rng = random.Random(42)
    orders = ['random', 'ascending', 'descending', 'sparse_first', 'prime_first',
              'random', 'sparse_first', 'random']

    for order in orders:
        if time.time() - start_time > timeout - 40:
            break
        chosen_arr, counts, in_set = greedy_random(rng, order)
        if len(chosen_arr) > best_len:
            best_len = len(chosen_arr)
            best_set = chosen_arr.copy()

    # Phase 2: Local search with congestion on the best greedy solution
    if best_len > 0 and time.time() - start_time < timeout - 30:
        chosen_arr = best_set.copy()
        in_set = np.zeros(N + 1, dtype=bool)
        for x in chosen_arr:
            in_set[x] = True
        counts = build_counts(chosen_arr.tolist())
        chosen_arr, counts, in_set = local_search_congestion(
            chosen_arr, counts, in_set,
            max_rounds=100, start_time=start_time, timeout=timeout - 20
        )
        if len(chosen_arr) > best_len:
            best_len = len(chosen_arr)
            best_set = chosen_arr.copy()

    # Phase 3: SA polish
    if best_len > 0 and time.time() - start_time < timeout - 15:
        chosen_arr = best_set.copy()
        in_set = np.zeros(N + 1, dtype=bool)
        for x in chosen_arr:
            in_set[x] = True
        counts = build_counts(chosen_arr.tolist())
        chosen_arr, counts, in_set = local_search_sa(
            chosen_arr, counts, in_set,
            start_time=start_time, timeout=timeout - 10
        )
        if len(chosen_arr) > best_len:
            best_len = len(chosen_arr)
            best_set = chosen_arr.copy()

    # Phase 4: More restarts with local search if time allows
    while time.time() - start_time < timeout - 8:
        order = random.choice(['random', 'sparse_first', 'prime_first', 'ascending'])
        chosen_arr, counts, in_set = greedy_random(random.Random(), order)

        remaining = timeout - (time.time() - start_time) - 4
        if remaining > 3:
            chosen_arr, counts, in_set = local_search_congestion(
                chosen_arr, counts, in_set,
                max_rounds=40, start_time=start_time, timeout=timeout - 4
            )

        if len(chosen_arr) > best_len:
            best_len = len(chosen_arr)
            best_set = chosen_arr.copy()

    # Final: convert to list of ints
    return list(best_set.astype(int))
