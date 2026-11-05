"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s requirement

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    # Pre-compute bit masks for initialization
    NUM_WORDS = (1 << N) // 64  # 4096
    total_inputs = 1 << N  # 262144
    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    # Pre-compute initial wires: wire i has bit k set if input k has bit i set
    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    def verify_network(comparators):
        """
        Fast verification using bit-parallelism.
        Returns True if the network sorts all 2^N Boolean inputs.
        """
        wires = [w.copy() for w in initial_wires]
        bitwise_and = np.bitwise_and
        bitwise_or = np.bitwise_or
        bitwise_not = np.bitwise_not
        any_ = np.any

        for (i, j) in comparators:
            wi = wires[i]
            wj = wires[j]
            new_i = bitwise_and(wi, wj)
            new_j = bitwise_or(wi, wj)
            wires[i] = new_i
            wires[j] = new_j

        # Check that the output is sorted: wire k should be subset of wire k+1
        for k in range(N - 1):
            if any_(wires[k] & bitwise_not(wires[k + 1])):
                return False
        return True

    # Removed redundant verify_network_fast, using verify_network

    def greedy_remove(comparators, time_limit=None):
        """
        Try to remove comparators one by one, scanning from the end.
        Multiple passes until no more removals possible or time limit hit.
        """
        current = list(comparators)
        if time_limit is None:
            time_limit = deadline - 1.0

        while True:
            if time.time() > time_limit:
                break
            removed = False
            for idx in range(len(current) - 1, -1, -1):
                if time.time() > time_limit:
                    return current
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    removed = True
                    break
            if not removed:
                break
        return current

    def local_search(comparators, time_limit=None):
        """
        Perform local search by replacing and removing comparators.
        Uses random restarts and diverse moves.
        """
        if time_limit is None:
            time_limit = deadline - 2.0

        current = list(comparators)
        rng = random.Random(42)

        best = current
        best_count = len(current)

        while time.time() < time_limit:
            if not current:
                break

            # Choose a random index to modify
            idx = rng.randint(0, len(current) - 1)
            old_i, old_j = current[idx]

            found_improvement = False

            # Try several random replacements
            for attempt in range(5):
                if time.time() > time_limit:
                    break

                # Generate a new comparator
                new_i = rng.randint(0, N - 1)
                new_j = rng.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i == new_j:
                    continue
                if (new_i, new_j) == (old_i, old_j):
                    continue

                test = list(current)
                test[idx] = (new_i, new_j)

                if verify_network(test):
                    current = test
                    found_improvement = True
                    break

            # If we made a replacement, try to remove some comparators
            if found_improvement:
                current = greedy_remove(current, time_limit)
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(current)

            # Occasionally try a pure removal pass
            if rng.random() < 0.3:
                current = greedy_remove(current, time_limit)
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(current)

            # Occasionally restart from best
            if rng.random() < 0.1 and len(current) > best_count:
                current = list(best)

        return best

    def hill_climb_remove_and_swap(comparators, time_limit=None):
        """
        More aggressive local search: try swapping adjacent comparators,
        removing, and replacing with better candidates.
        """
        if time_limit is None:
            time_limit = deadline - 2.0

        current = list(comparators)
        rng = random.Random(123)
        best = current
        best_count = len(current)

        while time.time() < time_limit:
            if not current:
                break

            # Try different move types
            move_type = rng.random()
            if move_type < 0.4:
                # Try removing a random comparator
                idx = rng.randint(0, len(current) - 1)
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    if len(current) < best_count:
                        best = list(current)
                        best_count = len(current)
                        current = greedy_remove(current, time_limit)
                        if len(current) < best_count:
                            best = list(current)
                            best_count = len(current)

            elif move_type < 0.7:
                # Try replacing a random comparator
                idx = rng.randint(0, len(current) - 1)
                old_i, old_j = current[idx]
                for _ in range(3):
                    if time.time() > time_limit:
                        break
                    new_i = rng.randint(0, N - 1)
                    new_j = rng.randint(0, N - 1)
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    if new_i == new_j or (new_i, new_j) == (old_i, old_j):
                        continue
                    test = list(current)
                    test[idx] = (new_i, new_j)
                    if verify_network(test):
                        current = test
                        if len(current) < best_count:
                            best = list(current)
                            best_count = len(current)
                        current = greedy_remove(current, time_limit)
                        if len(current) < best_count:
                            best = list(current)
                            best_count = len(current)
                        break

            else:
                # Try swapping two adjacent comparators
                if len(current) >= 2:
                    idx = rng.randint(0, len(current) - 2)
                    test = list(current)
                    test[idx], test[idx + 1] = test[idx + 1], test[idx]
                    if verify_network(test):
                        current = test
                        current = greedy_remove(current, time_limit)
                        if len(current) < best_count:
                            best = list(current)
                            best_count = len(current)

        return best

    # Start with odd-even transposition
    best_network = odd_even_transposition()
    best_count = len(best_network)

    # Phase 1: Greedy removal (this should get us close to 77)
    current = greedy_remove(best_network, deadline - 10.0)
    if len(current) < best_count:
        best_network = current
        best_count = len(best_network)

    # Phase 2: Local search with diverse moves
    current = local_search(best_network, deadline - 5.0)
    if len(current) < best_count:
        best_network = current
        best_count = len(best_network)

    # Phase 3: More aggressive hill climbing
    current = hill_climb_remove_and_swap(best_network, deadline - 2.0)
    if len(current) < best_count:
        best_network = current
        best_count = len(best_network)

    # Final verification
    if not verify_network(best_network):
        best_network = odd_even_transposition()

    return best_network
# EVOLVE-BLOCK-END
