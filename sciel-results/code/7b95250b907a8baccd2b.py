"""Size-efficient sorting network on 18 wires using systematic comparator optimization with bit-parallel verification."""
N = 18
import time
import random
import numpy as np
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin

    # Pre-compute bit masks for verification
    total_inputs = 1 << N  # 262144
    NUM_WORDS = total_inputs // 64  # 4096

    indices = np.arange(total_inputs, dtype=np.uint32)
    weights = 1 << np.arange(64, dtype=np.uint64)

    # Initial wire states: wire i has bit i of each input
    initial_wires = [None] * N
    for i in range(N):
        bits_i = (indices >> i) & 1
        bits_i_64 = bits_i.astype(np.uint64)
        packed = (bits_i_64.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        initial_wires[i] = packed

    # Target sorted state: for input with k ones, wires 0..N-k-1 have 0, wires N-k..N-1 have 1
    # We can compute this: wire j should have 1 iff input has more than (N-1-j) ones
    # i.e., popcount(input) > N-1-j
    popcounts = np.array([bin(x).count('1') for x in range(total_inputs)], dtype=np.uint8)
    target_wires = [None] * N
    for j in range(N):
        # Wire j should be 1 for inputs with popcount > N-1-j
        threshold = N - 1 - j
        mask = (popcounts > threshold).astype(np.uint64)
        packed = (mask.reshape(NUM_WORDS, 64) * weights).sum(axis=1)
        target_wires[j] = packed

    def apply_comparator(wires, i, j):
        """Apply comparator (i,j) to wires, return new wires."""
        wi = wires[i]
        wj = wires[j]
        new_wires = list(wires)
        new_wires[i] = wi & wj
        new_wires[j] = wi | wj
        return new_wires

    def verify_network(comparators):
        """Fast verification using bit-parallelism."""
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

    def count_violations(wires):
        """Count total bit-level violations in the wire state."""
        total = 0
        for k in range(N - 1):
            bad = wires[k] & ~wires[k + 1]
            total += int(np.sum(bad != 0))
        return total

    def odd_even_transposition():
        comparators = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comparators.append((i, i + 1))
        return comparators

    def batcher_odd_even_merge():
        """Batcher's odd-even merge sort network."""
        def merge(a, b):
            """Merge two sorted halves of size a+b into one sorted sequence."""
            n = a + b
            if n <= 1:
                return []
            comparators = []
            # Odd-even merge
            for i in range(n // 2):
                if i % 2 == 0:
                    comparators.append((i, i + n // 2))
                else:
                    comparators.append((i + n // 2 - 1, i))
            # Recursively merge odd and even subsequences
            odd_indices = list(range(1, n, 2))
            even_indices = list(range(0, n, 2))
            # This is a simplification; proper Batcher requires careful indexing
            # For now, use a simpler approach
            return comparators

        # Use a simpler known-good network
        return odd_even_transposition()

    def greedy_remove(comparators, time_limit=None):
        """Iteratively remove redundant comparators using systematic search."""
        current = list(comparators)
        changed = True
        while changed:
            changed = False
            if time_limit and time.time() > time_limit:
                break
            n = len(current)
            if n == 0:
                break

            # Try all positions (up to a limit for speed)
            max_checks = min(n, 100)
            positions = list(range(n))
            random.shuffle(positions)
            positions = positions[:max_checks]

            for idx in positions:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    def systematic_optimize(initial_net, deadline):
        """
        Systematic optimization: repeatedly try to remove comparators,
        and when stuck, try targeted insertions/replacements.
        """
        current = list(initial_net)
        best = list(current)
        best_len = len(current)

        # Phase 1: Aggressive greedy removal
        current = greedy_remove(current, time_limit=min(start_time + 5.0, deadline - 30.0))
        if len(current) < best_len:
            best = list(current)
            best_len = len(current)

        # Phase 2: Local search with targeted moves
        if time.time() > deadline - 10.0:
            return best

        # Precompute all possible comparators
        all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]

        for iteration in range(500):
            if time.time() > deadline - 5.0:
                break

            n = len(current)
            if n == 0:
                break

            # Try to remove a comparator
            if random.random() < 0.6:
                # Try removing a random comparator
                idx = random.randint(0, n - 1)
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
                    continue

            # Try replacing a comparator
            if random.random() < 0.8:
                idx = random.randint(0, n - 1)
                new_comp = random.choice(all_comparators)
                if new_comp == current[idx]:
                    continue
                test = current[:idx] + [new_comp] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
                    continue

            # Try inserting a comparator
            if n < 100:
                pos = random.randint(0, n)
                new_comp = random.choice(all_comparators)
                test = current[:pos] + [new_comp] + current[pos:]
                if verify_network(test):
                    current = test
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
                    continue

            # Try swapping two comparators
            if n >= 2:
                i1, i2 = random.sample(range(n), 2)
                test = current[:]
                test[i1], test[i2] = test[i2], test[i1]
                if verify_network(test):
                    current = test
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
                    continue

        return best

    # Generate multiple starting candidates
    candidates = []
    candidates.append(odd_even_transposition())

    # Add a few random dense networks as diverse starting points
    for seed in [1, 2, 3]:
        random.seed(seed)
        net = []
        for _ in range(150):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                net.append((i, j))
        random.shuffle(net)
        candidates.append(net)

    best_network = None
    best_count = float('inf')

    for idx, net in enumerate(candidates):
        if time.time() > deadline - 15.0:
            break

        # Prune the initial network
        prune_deadline = min(start_time + 4.0, deadline - 20.0)
        pruned = greedy_remove(net, time_limit=prune_deadline)

        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)

        # Run systematic optimization
        opt_deadline = deadline - 8.0
        opt_result = systematic_optimize(pruned, opt_deadline)

        if len(opt_result) < best_count:
            best_count = len(opt_result)
            best_network = list(opt_result)

        # Final prune
        if time.time() < deadline - 3.0:
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
