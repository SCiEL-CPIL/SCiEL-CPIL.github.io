"""Direct construction of a size-efficient sorting network on 18 wires using Simulated Annealing with error-guided repair."""
N = 18
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit, leave buffer)

    # Pre-compute bit masks for verification
    total_inputs = 1 << N  # 262144
    NUM_WORDS = total_inputs // 64  # 4096

    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all Boolean inputs.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    def find_violation(comparators):
        """
        Find a specific violation if the network is not a valid sorting network.
        Returns (violation_input, violating_pair) or (None, None) if valid.
        """
        wires = [w.copy() for w in initial_wires]

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            wires[i] = wi & wj
            wires[j] = wi | wj

        for k in range(N - 1):
            bad_mask = wires[k] & ~wires[k + 1]
            if np.any(bad_mask):
                # Find the first word with a violation
                word_idx = np.argmax(bad_mask)
                # Find the first bit in that word with a violation
                word_val = bad_mask[word_idx]
                bit_idx = 0
                while not (word_val & (1 << bit_idx)):
                    bit_idx += 1
                input_idx = word_idx * 64 + bit_idx
                return (input_idx, (k, k + 1))
        return (None, None)

    def greedy_remove(comparators, time_limit=None):
        """
        Iteratively remove redundant comparators.
        """
        current = list(comparators)
        changed = True
        while changed:
            changed = False
            if time_limit and time.time() > time_limit:
                break
            n = len(current)
            if n == 0:
                break

            if n <= 80:
                positions = list(range(n))
                random.shuffle(positions)
            else:
                positions = random.sample(range(n), min(80, n))

            for idx in positions:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    def bubble_sort_network():
        comparators = []
        for i in range(N - 1):
            for j in range(N - 1 - i):
                comparators.append((j, j + 1))
        return comparators

    def random_dense_network(num_comparators=200, seed=None):
        if seed is not None:
            random.seed(seed)
        comparators = []
        for _ in range(num_comparators):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                comparators.append((i, j))
        random.shuffle(comparators)
        return comparators

    def guided_mutation(candidate, violation_input, violating_pair, t):
        """
        Mutation that uses violation information to guide the repair.
        """
        result = list(candidate)
        v_input = violation_input
        v_pair = violating_pair
        
        # Determine which wires need to be compared to fix this violation
        # We need to bring the wire with 1 down and the wire with 0 up
        # Strategy: insert or modify comparators near the violation
        strategy = random.random()
        
        if strategy < 0.3 and len(result) > 0:
            # Replace a random comparator with one near the violation
            idx = random.randint(0, len(result) - 1)
            k, m = v_pair
            # Choose nearby wires to compare
            i_choice = max(0, k - random.randint(0, 2))
            j_choice = min(N - 1, m + random.randint(0, 2))
            if i_choice > j_choice:
                i_choice, j_choice = j_choice, i_choice
            if i_choice != j_choice:
                result[idx] = (i_choice, j_choice)
            else:
                # If invalid, just remove
                result.pop(idx)
        elif strategy < 0.5:
            # Insert a new comparator to help fix the violation
            k, m = v_pair
            # Insert near the end or middle
            pos = random.randint(0, len(result))
            i_choice = max(0, k - random.randint(0, 3))
            j_choice = min(N - 1, m + random.randint(0, 3))
            if i_choice > j_choice:
                i_choice, j_choice = j_choice, i_choice
            if i_choice != j_choice:
                result.insert(pos, (i_choice, j_choice))
        elif strategy < 0.7:
            # Remove a comparator (if we're too big)
            if len(result) > 0:
                idx = random.randint(0, len(result) - 1)
                result.pop(idx)
        elif strategy < 0.85:
            # Swap two comparators
            if len(result) >= 2:
                i1, i2 = random.sample(range(len(result)), 2)
                result[i1], result[i2] = result[i2], result[i1]
        else:
            # Random mutation
            if len(result) > 0 and random.random() < 0.5:
                idx = random.randint(0, len(result) - 1)
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u != v:
                    if u > v:
                        u, v = v, u
                    result[idx] = (u, v)
            else:
                if len(result) > 0:
                    idx = random.randint(0, len(result) - 1)
                    result.pop(idx)

        return result

    def simulated_annealing(initial_net, t_max=1000, t_min=0.5, alpha=0.997, max_iter=20000, deadline=None, seed=42):
        """
        Simulated Annealing with error-guided mutation.
        """
        current = list(initial_net)
        current_energy = len(current)
        best = list(current)
        best_energy = current_energy

        t = t_max
        random.seed(seed)
        
        # Check initial validity and get violation
        violation = find_violation(current)
        is_valid = violation[0] is None

        for i in range(max_iter):
            if deadline and time.time() > deadline:
                break

            if is_valid:
                # Standard SA: try to remove or modify
                r = random.random()
                candidate = list(current)

                if r < 0.55:
                    # Remove (higher probability)
                    if len(candidate) == 0:
                        continue
                    idx = random.randint(0, len(candidate) - 1)
                    candidate.pop(idx)
                elif r < 0.70:
                    # Insert
                    u = random.randint(0, N - 1)
                    v = random.randint(0, N - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    pos = random.randint(0, len(candidate))
                    candidate.insert(pos, (u, v))
                elif r < 0.85:
                    # Swap
                    if len(candidate) < 2:
                        continue
                    i1, i2 = random.sample(range(len(candidate)), 2)
                    candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                else:
                    # Replace
                    if len(candidate) == 0:
                        continue
                    idx = random.randint(0, len(candidate) - 1)
                    u = random.randint(0, N - 1)
                    v = random.randint(0, N - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    candidate[idx] = (u, v)
            else:
                # Guided mutation to fix the violation
                candidate = guided_mutation(current, violation[0], violation[1], t)
                if candidate == current:
                    # Try a different random mutation
                    candidate = list(current)
                    if len(candidate) > 0:
                        idx = random.randint(0, len(candidate) - 1)
                        u = random.randint(0, N - 1)
                        v = random.randint(0, N - 1)
                        if u != v:
                            if u > v:
                                u, v = v, u
                            candidate[idx] = (u, v)
                        else:
                            candidate.pop(idx)
                    else:
                        u = random.randint(0, N - 1)
                        v = random.randint(0, N - 1)
                        if u != v:
                            if u > v:
                                u, v = v, u
                            candidate.append((u, v))

            new_violation = find_violation(candidate)
            new_is_valid = new_violation[0] is None
            new_energy = len(candidate)

            if new_is_valid:
                if new_energy < current_energy:
                    # Always accept improvement
                    current = candidate
                    current_energy = new_energy
                    is_valid = True
                    violation = None
                    if new_energy < best_energy:
                        best = list(current)
                        best_energy = new_energy
                else:
                    # Accept with probability exp(-delta / T)
                    delta = new_energy - current_energy
                    if delta > 0:
                        if random.random() < np.exp(-delta / t):
                            current = candidate
                            current_energy = new_energy
                            is_valid = True
                            violation = None
            else:
                # Invalid: only accept if it's closer to valid (heuristic) or with small probability
                # Heuristic: accept if the number of violations decreased (not tracked, so just small prob)
                if random.random() < 0.05:  # Small chance to accept invalid states
                    current = candidate
                    current_energy = new_energy
                    is_valid = False
                    violation = new_violation
            t *= alpha
            if t < t_min:
                break

        return best

    # Generate starting candidates
    candidates = []
    candidates.append(("odd_even", odd_even_transposition()))
    candidates.append(("bubble", bubble_sort_network()))

    for i in range(5):
        density = [120, 140, 160, 180, 200][i]
        candidates.append((f"random_{i}", random_dense_network(density, seed=100 + i * 7)))

    best_network = None
    best_count = float('inf')

    for idx, (name, net) in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break

        # First, try to prune the initial network to get a better starting point for SA
        prune_time = min(start_time + 4.0, deadline - 12.0)
        pruned = greedy_remove(net, time_limit=prune_time)
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)

        remaining_time = deadline - 8.0 - time.time()
        if remaining_time < 5.0:
            break

        sa_result = simulated_annealing(
            pruned,
            t_max=500,
            t_min=0.5,
            alpha=0.997,
            max_iter=15000,
            deadline=deadline - 5.0,
            seed=42 + idx * 100
        )

        if len(sa_result) < best_count:
            best_count = len(sa_result)
            best_network = list(sa_result)

        final_pruned = greedy_remove(best_network, time_limit=deadline - 2.0)
        if len(final_pruned) < best_count:
            best_count = len(final_pruned)
            best_network = list(final_pruned)

    if best_network is None:
        best_network = odd_even_transposition()
        best_network = greedy_remove(best_network, time_limit=deadline - 1.0)

    if not verify_network(best_network):
        fallback = odd_even_transposition()
        fallback = greedy_remove(fallback, time_limit=deadline - 1.0)
        return fallback

    return best_network
