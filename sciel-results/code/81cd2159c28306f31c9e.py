"""Size-efficient sorting networks on 18 wires - Simulated Annealing with vectorized verification."""
import numpy as np
import time
import random

N = 18
NUM_BOOL_INPUTS = 1 << N  # 262144

# Pre-compute all Boolean inputs as a matrix of shape (262144, 18)
def _build_bool_matrix():
    """Build matrix where row k represents the k-th binary number with N bits."""
    bits = np.arange(NUM_BOOL_INPUTS, dtype=np.uint8).reshape(-1, 1)
    # Each wire gets one bit position
    matrix = np.zeros((NUM_BOOL_INPUTS, N), dtype=np.uint8)
    for w in range(N):
        matrix[:, w] = (bits >> w) & 1
    return matrix

_BOOL_MATRIX = None

def _get_bool_matrix():
    global _BOOL_MATRIX
    if _BOOL_MATRIX is None:
        _BOOL_MATRIX = _build_bool_matrix()
    return _BOOL_MATRIX


def _apply_comparator_batch(matrix, i, j):
    """Apply compare-exchange on wires i,j to a (262144, 18) uint8 matrix.
    min goes to wire i, max goes to wire j."""
    a = matrix[:, i].copy()
    b = matrix[:, j].copy()
    # min(a,b) to wire i, max(a,b) to wire j
    min_ab = np.minimum(a, b)
    max_ab = np.maximum(a, b)
    matrix[:, i] = min_ab
    matrix[:, j] = max_ab
    return matrix


def _verify_network(comparators):
    """Verify that the network sorts all 262144 Boolean inputs.
    Returns True if sorted, False otherwise."""
    mat = _get_bool_matrix().copy()
    for (i, j) in comparators:
        _apply_comparator_batch(mat, i, j)
    # Check: output should be sorted - all 1s come before all 0s on each wire
    # Actually, sorted means: for the output, wire 0 <= wire 1 <= ... <= wire N-1
    # For Boolean inputs, sorted means there's a threshold: wires 0..k-1 are 0, wires k..N-1 are 1
    # So we need: mat[:, 0] <= mat[:, 1] <= ... <= mat[:, N-1]
    # Equivalently: for each row, the sequence is non-decreasing
    # For Boolean: sum of ones should equal the input sum (conservation) AND be contiguous at the end
    
    # Check non-decreasing: mat[:, w] <= mat[:, w+1] for all w
    diffs = np.diff(mat, axis=1)
    if np.any(diffs < 0):
        return False
    return True


def _count_comparators(comparators):
    return len(comparators)


def _compute_depth(comparators):
    """Compute parallel depth of the network."""
    if not comparators:
        return 0
    # depth[w] = number of comparators wire w has been involved in sequentially
    depth = np.zeros(N, dtype=int)
    max_depth = 0
    for (i, j) in comparators:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
        max_depth = max(max_depth, d)
    return max_depth


def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _mutate(comparators, rng):
    """Apply a random mutation to the comparator list. Returns new list."""
    comps = list(comparators)
    op = rng.random()
    
    if op < 0.40 and len(comps) > 1:
        # Remove a random comparator
        idx = rng.randrange(len(comps))
        comps.pop(idx)
    elif op < 0.60:
        # Swap two comparators (change order)
        if len(comps) < 2:
            return comps
        i, j = rng.sample(range(len(comps)), 2)
        comps[i], comps[j] = comps[j], comps[i]
    elif op < 0.80:
        # Change a comparator to a random different pair
        if len(comps) < 1:
            return comps
        idx = rng.randrange(len(comps))
        i, j = rng.sample(range(N), 2)
        if i > j:
            i, j = j, i
        comps[idx] = (i, j)
    else:
        # Insert a random comparator at a random position
        i, j = rng.sample(range(N), 2)
        if i > j:
            i, j = j, i
        pos = rng.randrange(len(comps) + 1)
        comps.insert(pos, (i, j))
    
    return comps


def construct():
    """Construct a sorting network for 18 wires, minimizing comparator count."""
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 60s limit
    
    rng = random.Random(42)
    
    # Start from seed
    current = _seed_network()
    best = list(current)
    best_count = len(best)
    
    # Verify seed is correct
    if not _verify_network(current):
        # Fallback: just return seed
        return current
    
    # Simulated Annealing
    temp = 30.0  # initial temperature
    cooling = 0.995
    
    while time.time() - start_time < DEADLINE:
        new_comps = _mutate(current, rng)
        new_count = len(new_comps)
        
        # Check if valid
        is_valid = _verify_network(new_comps)
        
        if is_valid:
            if new_count < best_count:
                best = list(new_comps)
                best_count = new_count
                current = list(new_comps)
                # Boost temperature slightly when finding improvement
                temp = max(temp, 20.0)
            elif new_count == best_count:
                current = list(new_comps)
            else:
                # Worse but valid: accept with probability exp(-delta/temp)
                delta = new_count - best_count
                if delta > 0:
                    prob = np.exp(-delta / max(temp, 0.1))
                    if rng.random() < prob:
                        current = list(new_comps)
                    else:
                        # Reject
                        pass
                else:
                    current = list(new_comps)
        else:
            # Invalid: reject, but occasionally accept to escape local minima
            # Only accept invalid if we're stuck (high temp)
            if temp > 50.0 and rng.random() < 0.05:
                current = list(new_comps)
            # Otherwise reject
        
        # Cool down
        temp *= cooling
        temp = max(temp, 0.5)
    
    return best


# EVOLVE-BLOCK-START
# (The construct function above is the entry point)
# EVOLVE-BLOCK-END
