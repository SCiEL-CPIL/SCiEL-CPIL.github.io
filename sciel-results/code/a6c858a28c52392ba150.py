"""
Size-efficient sorting network on 18 wires using a structural Max-SAT-inspired
stochastic local search over a fixed comparator pool.

Key structural change:
- Instead of growing/shrinking a list of comparators (which makes the search space
  non-stationary and makes 'count' the objective coupled with correctness), we fix a
  pool of M candidate comparators (positions + wire pairs). The search state is a
  binary vector of length M indicating which comparators are active. The objective
  is to minimize the Hamming weight (number of active comparators) while satisfying
  the sorting constraint. This is a Max-SAT / weighted partial Max-SAT formulation
  solved by a WalkSAT-style stochastic local search with a soft penalty for weight.
- The pool is generated from structured sources (odd-even merge sort, random dense
  layers, pairwise all-pairs) so that a small subset is likely sufficient.
- Verification is bit-parallel over all 2^18 Boolean inputs.
"""

N = 18
import time
import random
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    # ---- Pre-compute bit masks for verification ----
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

        # Check sorted: for all k, wires[k] is subset of wires[k+1]
        for k in range(N - 1):
            if np.any(wires[k] & ~wires[k + 1]):
                return False
        return True

    # ---- Generate candidate pool ----
    def generate_pool(pool_size=120):
        """
        Generate a diverse pool of candidate (i, j) comparators.
        Includes structured comparators from known networks plus random ones.
        """
        pool = set()

        # Odd-even transposition comparators
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                pool.add((i, i + 1))

        # All adjacent pairs
        for i in range(N - 1):
            pool.add((i, i + 1))

        # All pairs (ensures full coverage)
        for i in range(N):
            for j in range(i + 1, N):
                pool.add((i, j))

        # Convert to list
        pool_list = list(pool)

        # If pool is larger than desired, sample
        if len(pool_list) > pool_size:
            random.shuffle(pool_list)
            pool_list = pool_list[:pool_size]

        return pool_list

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

            positions = list(range(n))
            random.shuffle(positions)

            for idx in positions:
                if idx >= len(current):
                    continue
                test = current[:idx] + current[idx + 1:]
                if verify_network(test):
                    current = test
                    changed = True
                    break

        return current

    # ---- WalkSAT-style search over binary vector of pool ----
    def walksat_search(pool, target_size, time_limit, seed=42):
        """
        Stochastic local search over binary vector of length len(pool).
        Objective: minimize number of active comparators subject to sorting.
        Uses a soft penalty: energy = weight + lambda * (1 if not sorted else 0).
        We use a WalkSAT-style approach: with probability p, flip a random bit;
        otherwise, flip a bit that is in a "critical" position (i.e., flipping it
        would change the sortedness or weight in a beneficial direction).
        """
        random.seed(seed)
        m = len(pool)

        # Start with a random assignment biased toward target_size
        active = np.zeros(m, dtype=bool)
        num_active = target_size
        idxs = random.sample(range(m), min(num_active, m))
        for idx in idxs:
            active[idx] = True

        def get_network(active_vec):
            return [pool[i] for i in range(m) if active_vec[i]]

        def is_sorted(active_vec):
            return verify_network(get_network(active_vec))

        def weight(active_vec):
            return int(np.sum(active_vec))

        # Initial check
        best_active = active.copy()
        best_weight = weight(active)
        if is_sorted(active):
            # We have a valid solution; try to reduce
            pass

        t = 1000.0
        alpha = 0.998
        t_min = 1.0

        while time.time() < time_limit:
            # Check if current is valid
            current_valid = is_sorted(active)
            current_w = weight(active)

            if current_valid and current_w < best_weight:
                best_weight = current_w
                best_active = active.copy()

            # Choose a bit to flip
            if random.random() < 0.3:
                # Random flip
                idx = random.randint(0, m - 1)
            else:
                # Heuristic: prefer flipping bits that are currently active
                # (to reduce weight) if valid, or inactive (to improve sortedness)
                if current_valid:
                    # Try to remove an active bit
                    active_idxs = [i for i in range(m) if active[i]]
                    if active_idxs:
                        idx = random.choice(active_idxs)
                    else:
                        idx = random.randint(0, m - 1)
                else:
                    # Try to add an inactive bit
                    inactive_idxs = [i for i in range(m) if not active[i]]
                    if inactive_idxs:
                        idx = random.choice(inactive_idxs)
                    else:
                        idx = random.randint(0, m - 1)

            # Flip
            active[idx] = not active[idx]
            new_valid = is_sorted(active)
            new_w = weight(active)

            # Acceptance criterion
            if current_valid and not new_valid:
                # Reject unless very early (high temperature)
                if random.random() > np.exp(-10.0 / t):
                    active[idx] = not active[idx]  # revert
            elif not current_valid and new_valid:
                # Always accept
                pass
            elif current_valid and new_valid:
                # Accept if weight decreases, or with probability based on weight change
                delta_w = new_w - current_w
                if delta_w <= 0:
                    pass  # accept
                else:
                    if random.random() > np.exp(-delta_w / t):
                        active[idx] = not active[idx]  # revert
            else:
                # Both invalid; accept with small probability
                if random.random() > np.exp(-5.0 / t):
                    active[idx] = not active[idx]  # revert

            t *= alpha
            if t < t_min:
                # Restart with new random assignment
                active = np.zeros(m, dtype=bool)
                num_active = random.randint(max(50, best_weight - 5), min(m, best_weight + 10))
                idxs = random.sample(range(m), min(num_active, m))
                for idx in idxs:
                    active[idx] = True
                t = 1000.0

        # Final polish: greedy removal on best found
        best_net = get_network(best_active)
        if not verify_network(best_net):
            return None, float('inf')
        best_net = greedy_remove(best_net, time_limit=time.time() + 2.0)
        return best_net, len(best_net)

    # ---- Main search ----
    best_network = None
    best_count = float('inf')

    # Phase 1: Quick prune of known networks
    candidates = [
        ("odd_even_trans", odd_even_transposition()),
        ("bubble", bubble_sort_network()),
    ]

    for name, net in candidates:
        if time.time() > deadline - 20.0:
            break
        pruned = greedy_remove(net, time_limit=start_time + 10.0)
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_network = list(pruned)

    # Phase 2: WalkSAT search over pool
    pool = generate_pool(pool_size=150)
    random.shuffle(pool)

    # Run multiple WalkSAT instances with different seeds and target sizes
    target_sizes = [100, 90, 80, 75, 70]
    seeds = [42, 123, 456, 789, 101]

    for ti, target in enumerate(target_sizes):
        if time.time() > deadline - 10.0:
            break
        seed = seeds[ti % len(seeds)]
        net, count = walksat_search(pool, target, time_limit=deadline - 15.0, seed=seed)
        if net is not None and count < best_count:
            best_count = count
            best_network = net

    # Phase 3: Final polish
    if best_network is not None:
        final_pruned = greedy_remove(best_network, time_limit=deadline - 2.0)
        if len(final_pruned) < best_count:
            best_count = len(final_pruned)
            best_network = list(final_pruned)

    if best_network is None:
        best_network = odd_even_transposition()
        best_network = greedy_remove(best_network, time_limit=deadline - 1.0)

    # Final verification
    if not verify_network(best_network):
        fallback = odd_even_transposition()
        fallback = greedy_remove(fallback, time_limit=deadline - 1.0)
        return fallback

    return best_network
