"""Size-efficient sorting networks on 18 wires via simulated annealing with Replace moves."""
import time
import random
import numpy as np

N = 18
# All possible comparator pairs (i, j) with i < j
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
NUM_PAIRS = len(ALL_PAIRS)

# Precompute boolean test vectors: all 2^18 inputs
# We'll use a compact representation: for each input, a 18-bit integer
# To speed up verification, we can use the zero-one principle: only need to test 2^18 = 262144 inputs
# But we can be smarter: use the "signature" method or batch processing

# For fast sorting network verification, we simulate the network on all 2^N inputs simultaneously using bit-packing.
# Each input is an 18-bit value. We represent the state of all 262144 inputs as an array of 18 numpy arrays (one per wire),
# each of length 262144, containing 0s and 1s.

def _build_initial_network():
    """Odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _verify_network(comps):
    """
    Verify that the sorting network sorts all 2^N inputs.
    Returns True if valid, False otherwise.
    Uses the zero-one principle: test all 2^N binary inputs.
    """
    # For each of the 2^N inputs, we have a vector of N bits.
    # We'll represent all inputs as an array of shape (2^N, N) of uint8.
    # But 2^18 * 18 = ~4.7M bytes, which is fine.
    
    # Actually, let's use a more memory-efficient approach:
    # Process inputs in batches, or use the fact that we can represent
    # all 2^18 inputs as their binary representations.
    
    # Generate all 2^18 inputs as an array of shape (262144, 18)
    # Input k (0 to 262143) has bits: bit i of k is the value on wire i
    # Wait, we need to be careful about bit ordering.
    
    # Let's define: input vector x where x[i] = bit i of the input integer
    # For input value v (0 to 2^18-1), wire i gets (v >> i) & 1
    
    # After sorting, the output should be sorted: the number of 1s is preserved,
    # and the 1s are in the highest-numbered wires (since we sort ascending: min to lower wire).
    # Actually, compare-exchange writes min to wire i (lower index) and max to wire j (higher index).
    # So the sorted order is: smallest values on lowest wires.
    # For a binary input with k ones, the sorted output has 1s on wires (N-k) through (N-1).
    
    # Let's build the input matrix.
    # To save memory and time, we can process in chunks.
    # But 262144 * 18 * 1 byte = ~4.7 MB, which is fine.
    
    num_inputs = 1 << N
    # Create array of shape (num_inputs, N), dtype uint8
    # inputs[i, j] = bit j of i
    # We can create this efficiently:
    # For bit j, the pattern is: 0 for 2^j entries, 1 for 2^j entries, repeating.
    
    # Let's create the full input matrix
    # Actually, for speed, let's use numpy broadcasting or precomputed patterns.
    
    # Alternative: we can verify by checking that for each input, the output is sorted.
    # A binary vector is sorted (ascending) iff all 0s come before all 1s,
    # i.e., once we see a 1, all subsequent entries must be 1.
    # Equivalently, the output is sorted iff the number of 1s in the first k wires
    # equals min(k, total_ones) for all k... no, that's not right.
    # For ascending sort: wire 0 has smallest, wire N-1 has largest.
    # So sorted binary vector: 0, 0, ..., 0, 1, 1, ..., 1
    # Where the number of 1s is preserved.
    
    # So for an input with `ones` ones, the sorted output has:
    # wires 0 to N-1-ones: 0
    # wires N-ones to N-1: 1
    
    # Let's build the input matrix efficiently.
    # Create an array of shape (num_inputs,) with values 0 to num_inputs-1
    # Then for each wire j, the value is (arr >> j) & 1
    
    # To avoid creating the full (262144, 18) array at once (which is fine but let's be careful),
    # let's just do it.
    
    # Actually, let's use a different approach for speed:
    # We'll simulate the network on all inputs in parallel using bit operations.
    # But numpy is fine for 262144 x 18.
    
    # Create the input matrix
    # input_vals = np.arange(num_inputs, dtype=np.uint32)
    # state = np.zeros((num_inputs, N), dtype=np.uint8)
    # for j in range(N):
    #     state[:, j] = (input_vals >> j) & 1
    
    # This might be slow in a loop. Let's vectorize.
    # state[:, j] = (input_vals >> j) & 1
    # We can do:
    # shifts = np.arange(N, dtype=np.uint32)
    # state = ((input_vals[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    
    # But input_vals[:, None] is (262144, 1), shifts[None, :] is (1, 18), result is (262144, 18)
    # This creates a large temporary array. 262144 * 18 * 4 bytes = ~18 MB for uint32 intermediate.
    # That's acceptable.
    
    input_vals = np.arange(num_inputs, dtype=np.uint32)
    shifts = np.arange(N, dtype=np.uint32)
    state = ((input_vals[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    
    # Now simulate the network
    for (i, j) in comps:
        # Compare-exchange: min to wire i, max to wire j
        # For each input, swap if state[_, i] > state[_, j]
        # Since values are 0 or 1:
        # if state[_, i] == 1 and state[_, j] == 0, swap them
        # After: state[_, i] = min(state[_, i], state[_, j]), state[_, j] = max(...)
        
        # For binary: min = AND, max = OR
        # But only if we swap when out of order. Actually:
        # min(a,b) and max(a,b) for binary:
        # (0,0)->(0,0), (0,1)->(0,1), (1,0)->(0,1), (1,1)->(1,1)
        # So: new_i = a AND b, new_j = a OR b
        # Wait, that's only correct if we always put min in i and max in j.
        # min(0,1)=0, max(0,1)=1 -> (0,1) correct
        # min(1,0)=0, max(1,0)=1 -> (0,1) correct, and we swapped
        # Yes, so new_i = a & b, new_j = a | b works for binary.
        
        a = state[:, i].copy()
        b = state[:, j].copy()
        state[:, i] = a & b
        state[:, j] = a | b
    
    # Check if all outputs are sorted
    # For each input, the output should be: 0s then 1s
    # Equivalent: for each input, the output is non-decreasing
    # i.e., state[k, i] <= state[k, i+1] for all i
    # Or: no pattern (1, 0) in adjacent wires
    
    # Check: for each adjacent pair, no (1, 0)
    # violations = (state[:, :-1] == 1) & (state[:, 1:] == 0)
    # return not np.any(violations)
    
    # More efficient: check that the output is sorted
    # A binary vector is sorted iff the number of transitions from 0 to 1 is at most 1,
    # and no transition from 1 to 0.
    # So check: no (1, 0) adjacent pair.
    
    # Let's compute: for each input, is there any i such that state[i]=1 and state[i+1]=0?
    # We can do:
    # left_ones = state[:, :-1]  # shape (num_inputs, N-1)
    # right_zeros = (state[:, 1:] == 0)
    # violations = left_ones & right_zeros
    # return not np.any(violations)
    
    # Even faster: sum up violations
    # Actually, np.any is fast enough.
    
    # But we can also check: the sorted output for an input with `ones` ones has
    # exactly `ones` ones in the last `ones` positions.
    # So: for each input, the number of ones in the first (N - ones) positions should be 0.
    # This is equivalent to the sorted check.
    
    # Let's just check adjacent pairs for (1, 0)
    # To avoid creating large temporaries, we can do:
    # for i in range(N-1):
    #     if np.any((state[:, i] == 1) & (state[:, i+1] == 0)):
    #         return False
    # return True
    
    # This is N-1 = 17 checks, each on 262144 elements. Should be fast.
    
    for i in range(N - 1):
        if np.any((state[:, i] == 1) & (state[:, i + 1] == 0)):
            return False
    return True

def _verify_network_fast(comps):
    """
    Faster verification using the zero-one principle but with early termination.
    We process inputs in batches and check as we go.
    """
    # For speed, let's use a different approach:
    # We only need to check inputs that are "critical".
    # But for simplicity, let's just do the full check but with numpy.
    
    # Actually, let's use a smarter method:
    # The zero-one principle says we only need to test 0/1 inputs.
    # There are 2^18 = 262144 such inputs.
    # We can represent the state of all inputs as 18 bit-arrays, each of length 262144.
    # But we already have a working method. Let's optimize it.
    
    # Let's use the fact that we can process all inputs at once with numpy.
    # The bottleneck is the 18 comparisons per comparator.
    
    # Alternative: use the "signature" approach where we track the set of inputs
    # that are still "unsorted" and stop early.
    
    # For now, let's use the full verification but make it as fast as possible.
    
    num_inputs = 1 << N
    input_vals = np.arange(num_inputs, dtype=np.uint32)
    shifts = np.arange(N, dtype=np.uint32)
    state = ((input_vals[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    
    for (i, j) in comps:
        a = state[:, i]
        b = state[:, j]
        # min and max for binary
        state[:, i] = a & b
        state[:, j] = a | b
    
    # Check sorted
    for i in range(N - 1):
        if np.any((state[:, i] == 1) & (state[:, i + 1] == 0)):
            return False
    return True

# For even faster verification, let's precompute the input matrix once and reuse it.
# But the state changes, so we need to copy it each time.

# Let's create a cached version of the input matrix
_INPUT_MATRIX = None

def _get_input_matrix():
    global _INPUT_MATRIX
    if _INPUT_MATRIX is None:
        num_inputs = 1 << N
        input_vals = np.arange(num_inputs, dtype=np.uint32)
        shifts = np.arange(N, dtype=np.uint32)
        _INPUT_MATRIX = ((input_vals[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return _INPUT_MATRIX

def _verify_network_cached(comps):
    """Verify using cached input matrix."""
    state = _get_input_matrix().copy()
    
    for (i, j) in comps:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = a & b
        state[:, j] = a | b
    
    for i in range(N - 1):
        if np.any((state[:, i] == 1) & (state[:, i + 1] == 0)):
            return False
    return True

def _depth_of_network(comps):
    """Compute the parallel depth of the network."""
    # Depth is the length of the longest path in the dependency graph.
    # We can compute this by processing comparators in order and tracking
    # the "level" of each wire.
    levels = [0] * N
    for (i, j) in comps:
        # The comparator (i, j) can only execute after both wires i and j have been
        # processed up to their current levels.
        # The new level for both wires is max(levels[i], levels[j]) + 1
        new_level = max(levels[i], levels[j]) + 1
        levels[i] = new_level
        levels[j] = new_level
    return max(levels)

def construct():
    """
    Construct a sorting network for 18 wires using simulated annealing.
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    # Start with odd-even transposition network
    current = _build_initial_network()
    current_valid = _verify_network_cached(current)
    current_len = len(current)
    best = current[:]
    best_len = current_len
    
    # Simulated annealing parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    rng = random.Random(42)
    
    # Precompute all possible pairs for replacement
    all_pairs = ALL_PAIRS
    
    step = 0
    while time.time() < deadline:
        step += 1
        if T < T_min:
            T = T_min
        
        # Choose a random move
        move_type = rng.random()
        
        new_network = current[:]
        
        if move_type < 0.25 and len(new_network) > 68:
            # Remove a random comparator
            idx = rng.randint(0, len(new_network) - 1)
            new_network.pop(idx)
        elif move_type < 0.50 and len(new_network) < 200:
            # Add a random comparator
            pair = rng.choice(all_pairs)
            new_network.append(pair)
        elif move_type < 0.75 and len(new_network) >= 2:
            # Swap two comparators
            i1, i2 = rng.sample(range(len(new_network)), 2)
            new_network[i1], new_network[i2] = new_network[i2], new_network[i1]
        elif move_type < 0.90 and len(new_network) >= 2:
            # Move a comparator to a new position
            i1 = rng.randint(0, len(new_network) - 1)
            i2 = rng.randint(0, len(new_network) - 1)
            comp = new_network.pop(i1)
            new_network.insert(i2, comp)
        else:
            # Replace a comparator with a random one
            if len(new_network) == 0:
                continue
            idx = rng.randint(0, len(new_network) - 1)
            new_pair = rng.choice(all_pairs)
            new_network[idx] = new_pair
        
        # Verify the new network
        new_valid = _verify_network_cached(new_network)
        new_len = len(new_network)
        
        # Accept/reject
        if new_valid:
            # Valid networks are always accepted if they're shorter
            if new_len <= current_len:
                current = new_network
                current_len = new_len
                if new_len < best_len:
                    best = new_network[:]
                    best_len = new_len
            else:
                # Longer valid network: accept with probability based on T
                # We prefer shorter networks, so delta = new_len - current_len > 0
                # Accept with prob exp(-delta / T)
                delta = new_len - current_len
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current = new_network
                    current_len = new_len
        else:
            # Invalid network: reject unless we're at high temperature
            # and the length is much shorter (to allow exploration)
            # Actually, for sorting networks, invalid is bad. Let's reject.
            # But to escape local minima, we might accept invalid temporarily.
            # Let's reject for simplicity.
            pass
        
        # Cool down
        T *= cooling_rate
        
        # Check time
        if time.time() > deadline - 2.0:
            break
    
    return best
