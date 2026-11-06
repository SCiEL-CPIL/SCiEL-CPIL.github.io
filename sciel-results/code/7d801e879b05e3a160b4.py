import numpy as np
import time
import random

N = 16
TOTAL_INPUTS = 1 << N

# Pre-compute all boolean inputs for fast vectorized checking
# This is done once at module load or first call.
def get_inputs():
    # Use a global cache
    if not hasattr(get_inputs, 'cache'):
        # Generate all 2^16 bit patterns
        # np.arange(65536).reshape(65536, 1)
        # bits = (arr >> np.arange(N-1, -1, -1)) & 1
        # This creates a (65536, 16) array of 0s and 1s
        arr = np.arange(TOTAL_INPUTS, dtype=np.uint8).reshape(-1, 1)
        # We want bit 15 (MSB) to be wire 0? Or wire 15?
        # The task says "min to wire i, max to j".
        # The inputs are just sequences. The specific bit mapping doesn't matter for validity as long as we check all 2^N permutations.
        # Let's map bit k (0-indexed from LSB) to wire k.
        # So wire 0 is LSB, wire 15 is MSB.
        # arr is column vector.
        # shift = np.arange(N-1, -1, -1, dtype=np.uint8) # 15, 14, ..., 0
        # bits = (arr >> shift) & 1
        # This gives shape (65536, 16).
        shift = np.arange(N-1, -1, -1, dtype=np.uint8)
        bits = (arr >> shift) & 1
        get_inputs.cache = bits
    return get_inputs.cache

INPUTS = get_inputs()

def check_network(net):
    """
    Returns the number of invalid inputs (0 if valid).
    Uses vectorized NumPy operations.
    """
    state = INPUTS.copy()
    for i, j in net:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    # Check if sorted: state[:, k] <= state[:, k+1] for all k
    # If any pair is out of order, it's invalid.
    # We can check if the array is non-decreasing along axis=1.
    diffs = np.diff(state, axis=1)
    # If any diff < 0, it's not sorted.
    invalid_mask = np.any(diffs < 0, axis=1)
    return np.sum(invalid_mask)

def initial_network():
    """
    Generates a valid Odd-Even Transposition network (120 comparators).
    """
    net = []
    for k in range(N - 1):
        if k % 2 == 0:
            for i in range(0, N - 1, 2):
                net.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                net.append((i, i + 1))
    return net

def construct():
    """
    Constructs a sorting network using Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Start with a valid network
    current_net = initial_network()
    best_net = current_net.copy()
    best_len = len(current_net)
    
    # Initial fitness (validity check)
    invalid_count = check_network(current_net)
    current_fitness = len(current_net) + 1000 * invalid_count
    best_fitness = current_fitness
    
    # Simulated Annealing parameters
    T = 50.0
    cooling_rate = 0.999
    
    # Pre-compute all possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        if move_type < 0.8:
            # Deletion
            if len(current_net) == 0:
                continue
            idx = random.randint(0, len(current_net) - 1)
            new_net = current_net[:idx] + current_net[idx+1:]
        else:
            # Replacement (Change a comparator)
            if len(current_net) == 0:
                continue
            idx = random.randint(0, len(current_net) - 1)
            new_pair = random.choice(all_pairs)
            new_net = current_net.copy()
            new_net[idx] = new_pair
        
        # Evaluate new network
        new_invalid = check_network(new_net)
        new_len = len(new_net)
        new_fitness = new_len + 1000 * new_invalid
        
        # Acceptance probability
        delta_E = new_fitness - current_fitness
        if delta_E < 0 or random.random() < np.exp(-delta_E / T):
            current_net = new_net
            current_fitness = new_fitness
            
            # Update best if valid and shorter
            if new_invalid == 0 and new_len < best_len:
                best_net = new_net.copy()
                best_len = new_len
                best_fitness = new_fitness
        
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1
            
    return best_net
