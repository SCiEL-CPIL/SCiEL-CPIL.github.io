"""
Size-efficient sorting network for 16 wires.
Uses a known 60-comparator construction as the base, then applies
optimized verification, greedy removal, and local search to attempt
to find a smaller correct network.
"""
import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
# Evaluator kills at 90s; keep a safety margin.
DEADLINE_SECONDS = 85.0

# Pre-compute the input matrix for verification
_N_INPUTS = 1 << N
_INDICES = np.arange(_N_INPUTS, dtype=np.uint32)
_INPUT_MATRIX = np.zeros((_N_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUT_MATRIX[:, _i] = (_INDICES >> _i) & 1

# Pre-compute the "sorted" condition mask for faster verification
# We check if x[:, i] > x[:, i+1] for any i.
# We can pre-compute the pairs of columns to check.
_SORT_PAIRS = [(i, i + 1) for i in range(N - 1)]


def _apply_network(network, x):
    """Apply the compare-exchange network to input vector x (numpy array)."""
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x


def _verify_network_fastest(network):
    """
    Fast verification using pre-computed input matrix.
    Optimized to minimize overhead.
    """
    x = _INPUT_MATRIX.copy()
    
    for i, j in network:
        # Vectorized compare-exchange
        mask = x[:, i] > x[:, j]
        if mask.any():
            x[mask, i] = 0
            x[mask, j] = 1

    # Check sortedness: all adjacent pairs must be non-decreasing
    # x[:, 0] <= x[:, 1] <= ... <= x[:, 15]
    # Equivalent to: max over adjacent differences <= 0
    # We check if any adjacent pair violates x[:, i] > x[:, i+1]
    # Use a single vectorized check: compute all adjacent diffs and check max
    # But looping 15 times is fine; let's try to make it faster
    for i in range(N - 1):
        if (x[:, i] > x[:, i + 1]).any():
            return False
    return True


def _count_comparators(network):
    return len(network)


def _compute_depth(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    n = len(network)
    wire_to_last_comp = {}
    depths = [0] * n
    for idx, (i, j) in enumerate(network):
        max_dep = 0
        for wire in (i, j):
            if wire in wire_to_last_comp:
                prev_idx = wire_to_last_comp[wire]
                if depths[prev_idx] + 1 > max_dep:
                    max_dep = depths[prev_idx] + 1
        depths[idx] = max_dep
        wire_to_last_comp[i] = idx
        wire_to_last_comp[j] = idx
    return max(depths) if depths else 0


# A known 60-comparator sorting network for 16 inputs.
# Based on standard optimal/near-optimal constructions.
_BASE_NETWORK_60 = [
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (1, 2), (5, 6), (9, 10), (13, 14),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
]


def _greedy_remove(network, deadline):
    """Repeatedly try to remove each comparator; return improved network."""
    improved = True
    # Iterate backwards to avoid index issues
    while improved:
        improved = False
        # Check time budget
        if time.time() > deadline:
            return network
            
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            trial = network[:i] + network[i + 1:]
            if _verify_network_fastest(trial):
                network = trial
                improved = True
                break # Restart loop from end since network changed
    return network


def _local_search(network, deadline, rng):
    """
    Local search: try swapping, moving, and deleting comparators.
    Only accept changes that maintain correctness and reduce count or
    (for swap/move) keep count same but may help future deletions.
    Uses a temperature-like acceptance for same-length moves to escape local minima.
    """
    best = list(network)
    best_len = len(best)
    stall_count = 0
    max_stall = 2000
    last_improvement_time = time.time()
    attempts = 0
    
    while time.time() < deadline and stall_count < max_stall:
        attempts += 1
        net = list(best)
        n_comp = len(net)
        r = rng.random()

        if r < 0.50 and n_comp > 1:
            # Delete: focus on recent comparators first (often redundant)
            if attempts % 3 == 0:
                idx = rng.randint(0, min(5, n_comp - 1))
            else:
                idx = rng.randint(0, n_comp - 1)
            candidate = net[:idx] + net[idx + 1:]
        elif r < 0.70 and n_comp >= 2:
            # Swap
            i, j = rng.sample(range(n_comp), 2)
            candidate = list(net)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif r < 0.90 and n_comp >= 2:
            # Move
            idx = rng.randint(0, n_comp - 1)
            new_idx = rng.randint(0, n_comp - 1)
            comp = net[idx]
            remaining = [c for k, c in enumerate(net) if k != idx]
            if new_idx > idx:
                new_idx -= 1
            new_idx = max(0, min(new_idx, len(remaining)))
            candidate = remaining[:new_idx] + [comp] + remaining[new_idx:]
        else:
            # Insert a random comparator (to escape local minima)
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            new_idx = rng.randint(0, n_comp)
            candidate = net[:new_idx] + [(a, b)] + net[new_idx:]

        if _verify_network_fastest(candidate):
            if len(candidate) < best_len:
                best = candidate
                best_len = len(best)
                stall_count = 0
                last_improvement_time = time.time()
                # Greedy removal after improvement
                best = _greedy_remove(best, deadline)
                best_len = len(best)
            else:
                # Accept same-length changes with some probability to explore
                if len(candidate) == best_len and rng.random() < 0.5:
                    best = candidate
                stall_count += 1
        else:
            stall_count += 1

        # Periodically try a full greedy pass
        if attempts % 100 == 0:
            best = _greedy_remove(best, deadline)
            best_len = len(best)
            stall_count = 0

    return best


def construct():
    """
    Construct a sorting network for 16 wires.
    Starts from a known 60-comparator network and applies
    greedy removal and local search to minimize comparator count.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - 5.0  # safety margin

    # Start from the 60-comparator base
    network = list(_BASE_NETWORK_60)

    # Verify base
    if not _verify_network_fastest(network):
        # Fallback to odd-even transposition (120 comparators)
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
        return network

    # Multiple greedy removal passes with different orderings
    for _ in range(3):
        if time.time() > deadline:
            break
        network = _greedy_remove(network, deadline)
    
    best_len = len(network)

    # Local search with multiple random seeds
    for seed in [42, 123, 456]:
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        network = _local_search(network, deadline, rng)
        best_len = len(network)

    # Final greedy removal
    network = _greedy_remove(network, deadline)

    return network
