"""
Sorting network for 17 wires using Simulated Annealing with Add-Remove kicks.
Uses the zero-one principle (131072 Boolean inputs) for fast verification.
"""
import time
import random
import numpy as np

N = 17
NUM_BOOL_INPUTS = 1 << N  # 131072

# Precompute all boolean inputs as a matrix of shape (131072, 17)
# Each row is a binary vector
_BOOL_INPUTS = np.arange(NUM_BOOL_INPUTS, dtype=np.uint8).reshape(1, -1)
# Actually, let's build it properly: each input is an integer 0..2^17-1,
# and we extract bits.
def _build_bool_inputs():
    # For each of 2^17 inputs, create a 17-bit vector
    # input k: bit i is (k >> i) & 1
    # We want shape (131072, 17) where column i is bit i
    arr = np.arange(NUM_BOOL_INPUTS, dtype=np.uint32)
    # Reshape to (1, 131072), then broadcast bits
    bits = np.zeros((N, NUM_BOOL_INPUTS), dtype=np.uint8)
    for i in range(N):
        bits[i] = (arr >> i) & 1
    # Transpose to (131072, N)
    return bits.T

_BOOL_MATRIX = _build_bool_inputs()  # shape (131072, 17), uint8


def _verify_network(comparators):
    """
    Verify that the given list of comparators sorts all 17-input sequences
    using the zero-one principle.
    
    Returns True if valid, False otherwise.
    """
    if not comparators:
        # Empty network only valid if N <= 1
        return N <= 1
    
    # Make a copy of the boolean matrix to work on
    # We'll simulate the network on all inputs at once
    # Shape: (131072, 17)
    state = _BOOL_MATRIX.copy()
    
    for (i, j) in comparators:
        # For each input row, compare elements at positions i and j
        # min goes to i, max goes to j
        a = state[:, i]
        b = state[:, j]
        # min = (a+b - |a-b|) / 2, but since binary: min = a AND b, max = a OR b
        # Actually for binary: min(a,b) = a & b, max(a,b) = a | b
        state[:, i] = a & b
        state[:, j] = a | b
    
    # After sorting, each row should be sorted: 0s then 1s
    # A sorted binary vector has all 0s first, then all 1s
    # Check: for each row, once we see a 1, all subsequent must be 1
    # Equivalent: the row is non-decreasing
    # Check if each row is sorted
    # A row is sorted iff for all k from 0 to N-2: state[row, k] <= state[row, k+1]
    # For binary: this means no pattern 1,0 exists
    # We can check: for each adjacent pair, state[:, k] <= state[:, k+1]
    # i.e., NOT (state[:, k] == 1 AND state[:, k+1] == 0)
    
    for k in range(N - 1):
        if np.any((state[:, k] == 1) & (state[:, k+1] == 0)):
            return False
    
    return True


def _depth_of_network(comparators):
    """Compute the parallel depth of the network."""
    if not comparators:
        return 0
    # Depth = longest path in the dependency graph
    # Two comparators are dependent if they share a wire
    # We can compute depth by simulating: assign each comparator a level
    # level(c) = 1 + max(level(c') for c' that shares a wire with c and comes before c)
    # Actually, depth is the minimum number of parallel steps.
    # Greedy: assign levels greedily
    wire_last_level = [0] * N
    max_level = 0
    for (i, j) in comparators:
        # This comparator must come after any previous comparator on wire i or j
        lvl = max(wire_last_level[i], wire_last_level[j]) + 1
        wire_last_level[i] = lvl
        wire_last_level[j] = lvl
        max_level = max(max_level, lvl)
    return max_level


def construct():
    """
    Construct a sorting network for 17 wires.
    Returns a list of (i, j) pairs.
    """
    start_time = time.time()
    TIME_LIMIT = 85.0  # Safety margin from 90s
    
    # Start with odd-even transposition network
    def make_seed():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    network = make_seed()
    best_network = list(network)
    best_count = len(network)
    
    # Verify seed
    if not _verify_network(network):
        # Should not happen, but just in case
        return best_network
    
    # Simulated Annealing
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    rng = random.Random(42)
    
    def _try_move(network, move_type, rng):
        """
        Try a random move. Returns (new_network, valid) or (None, False).
        move_type: 'add', 'remove', 'swap'
        """
        n = len(network)
        
        if move_type == 'add':
            # Add a random comparator
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, n)
            new_net = network[:pos] + [(i, j)] + network[pos:]
            return new_net, True
        
        elif move_type == 'remove':
            if n <= 1:
                return None, False
            pos = rng.randint(0, n - 1)
            new_net = network[:pos] + network[pos+1:]
            return new_net, True
        
        elif move_type == 'swap':
            if n < 2:
                return None, False
            p = rng.randint(0, n - 2)
            q = rng.randint(p + 1, n - 1)
            new_net = list(network)
            new_net[p], new_net[q] = new_net[q], new_net[p]
            return new_net, True
        
        return None, False
    
    # Main SA loop
    iteration = 0
    while time.time() - start_time < TIME_LIMIT:
        iteration += 1
        
        # Choose move type
        # Bias towards removal when we're doing well
        r = rng.random()
        if r < 0.35:
            move_type = 'remove'
        elif r < 0.65:
            move_type = 'add'
        else:
            move_type = 'swap'
        
        new_network, ok = _try_move(network, move_type, rng)
        if not ok:
            continue
        
        new_count = len(new_network)
        
        # Verify
        is_valid = _verify_network(new_network)
        
        if is_valid:
            if new_count < best_count:
                best_network = list(new_network)
                best_count = new_count
            # Always accept valid moves
            network = new_network
        else:
            # Invalid: accept with probability based on temperature and penalty
            # Penalty = how many unsorted patterns exist (approximate by count)
            # For simplicity, use a fixed penalty
            penalty = 100.0  # Large penalty for invalid
            delta = penalty  # Cost increase
            if rng.random() < np.exp(-delta / T):
                network = new_network
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Periodic restart to escape local minima
        if iteration % 5000 == 0 and T <= T_min * 1.1:
            # Add-Remove kick: add several random comparators then re-optimize
            kick_network = list(network)
            for _ in range(3):
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                pos = rng.randint(0, len(kick_network))
                kick_network = kick_network[:pos] + [(i, j)] + kick_network[pos:]
            # Reset temperature
            T = 0.5
            network = kick_network
    
    # Final polish: try to remove any redundant comparators
    # Greedy removal pass
    polished = list(best_network)
    improved = True
    while improved:
        improved = False
        for idx in range(len(polished) - 1, -1, -1):
            candidate = polished[:idx] + polished[idx+1:]
            if _verify_network(candidate):
                polished = candidate
                improved = True
                break
        # Check time
        if time.time() - start_time > TIME_LIMIT - 5:
            break
    
    # Use the best of polished and best_network
    if len(polished) <= len(best_network):
        return polished
    else:
        return best_network
