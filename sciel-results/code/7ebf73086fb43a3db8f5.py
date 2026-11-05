"""Size-efficient sorting networks on 17 wires.

This program searches for a small comparator network that sorts all 17-input
sequences using the zero-one principle. It uses a randomized hill-climbing /
simulated-annealing style search over the space of comparator sequences,
verifying correctness via exhaustive Boolean-input simulation with NumPy
vectorization.

The search starts from the odd-even transposition network (136 comparators)
and attempts to reduce the comparator count by:
1. Removing redundant comparators (those whose removal still preserves sorting).
2. Swapping adjacent comparators to find better orderings that enable further removals.
3. Inserting new comparators at random positions to escape local minima.

A deadline guard ensures we return the best valid network found before the
90-second wall-clock limit.
"""

import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 90.0
SAFETY_MARGIN = 5.0  # seconds to spare


def _now():
    return time.time()


def _apply_comparator(wires, i, j):
    """Apply a single compare-exchange: min goes to wire i, max to wire j."""
    if wires[i] > wires[j]:
        wires[i], wires[j] = wires[j], wires[i]
    return wires


def _simulate_network(network, inputs):
    """
    Simulate the network on a batch of inputs (2D array, shape [num_inputs, N]).
    Returns the output array after all comparators.
    """
    result = inputs.copy()
    for i, j in network:
        # Compare-exchange on columns i and j
        col_i = result[:, i]
        col_j = result[:, j]
        # min and max
        min_val = np.minimum(col_i, col_j)
        max_val = np.maximum(col_i, col_j)
        result[:, i] = min_val
        result[:, j] = max_val
    return result


def _is_sorted_network(network, num_checks=None):
    """
    Verify that the network sorts all 2^N Boolean inputs using the zero-one principle.
    Uses vectorized NumPy simulation.
    
    Returns True if the network is a valid sorting network.
    """
    if num_checks is None:
        num_checks = 2 ** N  # full exhaustive check
    
    # Generate all 2^N Boolean inputs
    all_inputs = np.array(list(itertools.product([0, 1], repeat=N)), dtype=np.int8)
    
    # If we can't do full check (too many), sample
    if num_checks < 2 ** N:
        indices = np.random.choice(2 ** N, size=num_checks, replace=False)
        all_inputs = all_inputs[indices]
    
    output = _simulate_network(network, all_inputs)
    
    # Check if every row is sorted (non-decreasing)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True


def _is_sorted_network_fast(network):
    """
    Fast verification using the zero-one principle with vectorized operations.
    Processes all 2^17 = 131072 Boolean inputs at once.
    """
    # Generate all Boolean inputs as a 2D array
    # Use bit manipulation to generate all combinations
    all_inputs = np.zeros((2 ** N, N), dtype=np.int8)
    for k in range(N):
        # Bit k of each index
        all_inputs[:, k] = (np.arange(2 ** N, dtype=np.uint32) >> k) & 1
    
    output = _simulate_network(network, all_inputs)
    
    # Check sorted: for each adjacent pair, output[:,k] <= output[:,k+1]
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True


def _depth_of_network(network):
    """Compute the parallel depth of the network."""
    if not network:
        return 0
    depth = [0] * N
    max_depth = 0
    for i, j in network:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
        max_depth = max(max_depth, d)
    return max_depth


def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _random_comparator():
    """Generate a random valid comparator (i, j) with i < j."""
    i = random.randint(0, N - 1)
    j = random.randint(i + 1, N - 1)
    return (i, j)


def _remove_redundant(network):
    """
    Try to remove each comparator one at a time.
    Returns a new network with redundant comparators removed.
    """
    if not network:
        return network
    
    # Work on a copy
    net = list(network)
    changed = True
    while changed:
        changed = False
        # Try removing each comparator (from end to start to avoid index shifting issues)
        for idx in range(len(net) - 1, -1, -1):
            # Temporarily remove
            candidate = net[:idx] + net[idx + 1:]
            if _is_sorted_network_fast(candidate):
                net = candidate
                changed = True
                break  # Restart the scan since indices shifted
    return net


def construct():
    """
    Construct a sorting network on 17 wires with as few comparators as possible.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = _now()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    # Initialize with seed network
    best_network = _seed_network()
    best_count = len(best_network)
    
    # Verify seed
    if not _is_sorted_network_fast(best_network):
        # Should not happen, but just in case
        raise ValueError("Seed network is invalid")
    
    # Hill climbing with random restarts
    current_network = list(best_network)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    iteration = 0
    while _now() < deadline:
        iteration += 1
        
        # Check time more frequently
        if _now() > deadline - 2.0:
            break
        
        # Strategy: try to improve current network
        # 1. Try removing a random comparator
        if len(current_network) > 1:
            # Try removing a random comparator
            idx = random.randint(0, len(current_network) - 1)
            candidate = current_network[:idx] + current_network[idx + 1:]
            if _is_sorted_network_fast(candidate):
                current_network = candidate
                if len(current_network) < best_count:
                    best_network = list(current_network)
                    best_count = len(best_network)
                continue
        
        # 2. Try swapping two adjacent comparators
        if len(current_network) > 1:
            idx = random.randint(0, len(current_network) - 2)
            candidate = list(current_network)
            candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            if _is_sorted_network_fast(candidate):
                current_network = candidate
                if len(current_network) < best_count:
                    best_network = list(current_network)
                    best_count = len(best_network)
                continue
        
        # 3. Try replacing a random comparator with a different one
        if len(current_network) > 0:
            idx = random.randint(0, len(current_network) - 1)
            candidate = list(current_network)
            candidate[idx] = _random_comparator()
            if _is_sorted_network_fast(candidate):
                current_network = candidate
                if len(current_network) < best_count:
                    best_network = list(current_network)
                    best_count = len(best_network)
                continue
        
        # 4. Try inserting a new comparator at a random position
        #    (This increases count, but may enable future reductions)
        if len(current_network) < MAX_COMPARATORS and random.random() < 0.1:
            idx = random.randint(0, len(current_network))
            candidate = current_network[:idx] + [_random_comparator()] + current_network[idx:]
            if _is_sorted_network_fast(candidate):
                # Only accept if it doesn't worsen too much, or randomly
                if random.random() < 0.3:
                    current_network = candidate
                # Don't update best if it's larger
                continue
        
        # 5. Random restart from seed with small perturbations
        if random.random() < 0.01:
            current_network = list(best_network)
            # Apply a few random modifications
            for _ in range(random.randint(1, 3)):
                if current_network and random.random() < 0.5:
                    idx = random.randint(0, len(current_network) - 1)
                    current_network[idx] = _random_comparator()
                else:
                    idx = random.randint(0, len(current_network))
                    current_network.insert(idx, _random_comparator())
            if _is_sorted_network_fast(current_network):
                if len(current_network) < best_count:
                    best_network = list(current_network)
                    best_count = len(best_network)
    
    # Final: try to remove redundant comparators from best network
    if _now() < deadline - 1.0:
        final_network = _remove_redundant(best_network)
        if len(final_network) < best_count:
            best_network = final_network
            best_count = len(best_network)
    
    return best_network


# Import itertools for potential use
import itertools

# Pre-compute all Boolean inputs for faster repeated verification
_ALL_BOOLEAN_INPUTS = None

def _get_all_boolean_inputs():
    global _ALL_BOOLEAN_INPUTS
    if _ALL_BOOLEAN_INPUTS is None:
        _ALL_BOOLEAN_INPUTS = np.zeros((2 ** N, N), dtype=np.int8)
        for k in range(N):
            _ALL_BOOLEAN_INPUTS[:, k] = (np.arange(2 ** N, dtype=np.uint32) >> k) & 1
    return _ALL_BOOLEAN_INPUTS


def _is_sorted_network_fast(network):
    """
    Fast verification using precomputed Boolean inputs.
    """
    all_inputs = _get_all_boolean_inputs()
    output = _simulate_network(network, all_inputs)
    
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True
