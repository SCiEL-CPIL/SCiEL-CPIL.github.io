"""Size-efficient sorting network search for N=16 using simulated annealing."""
import time
import random

N = 16
MAX_TIME = 85.0  # safety margin before 90s kill
DEADLINE = time.time() + MAX_TIME

# Precompute all possible comparators
ALL_COMPARATORS = [(i, j) for i in range(N) for j in range(i + 1, N)]
ALL_COMPARATORS_SET = set(ALL_COMPARATORS)

def apply_comparator_to_state(state, c):
    """Apply comparator c=(i,j) to a 16-bit state. Returns new state."""
    i, j = c
    vi = (state >> i) & 1
    vj = (state >> j) & 1
    if vi > vj:
        # swap: set i to 0, j to 1
        state ^= (1 << i) | (1 << j)
    return state

def apply_network_to_state(state, network):
    """Apply entire network to a state. Returns final state."""
    for c in network:
        i, j = c
        vi = (state >> i) & 1
        vj = (state >> j) & 1
        if vi > vj:
            state ^= (1 << i) | (1 << j)
    return state

def check_network_fast(network):
    """
    Check if network sorts all 65536 inputs using zero-one principle.
    We only need to check inputs where wire 0 has value 1 (or any specific wire).
    Actually, by zero-one principle, we need to check all 2^16 inputs.
    But we can be smarter: check all inputs where exactly k bits are set, for k in 0..16.
    Actually the standard trick: check all inputs where bit 0 is 1 (2^15 inputs) plus the all-zeros case.
    But let's just check all 65536 inputs efficiently.
    """
    # Optimization: check all 2^16 inputs
    for s in range(65536):
        final = apply_network_to_state(s, network)
        # For sorted output, all 1s must be in the lowest wires
        # The number of 1s in final must equal number of 1s in s
        # And the 1s must be contiguous at the bottom
        num_ones = bin(s).count('1')
        # Expected sorted: ones in wires 0..num_ones-1
        expected = (1 << num_ones) - 1
        if final != expected:
            return False
    return True

def check_network_partial(network, sample_size=2048):
    """
    Faster partial check: sample random inputs.
    Returns (is_valid, failure_count) where failure_count is 0 if no failures in sample.
    """
    # Check all-zeros and all-ones first
    if apply_network_to_state(0, network) != 0:
        return False, 1
    if apply_network_to_state(65535, network) != 65535:
        return False, 1
    
    # Sample random inputs
    for _ in range(sample_size):
        s = random.getrandbits(16)
        final = apply_network_to_state(s, network)
        num_ones = bin(s).count('1')
        expected = (1 << num_ones) - 1
        if final != expected:
            return False, 1
    return True, 0

def is_valid_network(network):
    """Full check of network."""
    return check_network_fast(network)

def compute_depth(network):
    """Compute parallel depth of the network."""
    # depth[i] = earliest time wire i can be 'final'
    # Actually, depth is the minimum number of layers such that comparators can be grouped
    # into layers where no two comparators in the same layer share a wire.
    # Greedy: assign each comparator to the earliest layer where its wires are free.
    wire_layer = [0] * N
    depth = 0
    for c in network:
        i, j = c
        layer = max(wire_layer[i], wire_layer[j])
        wire_layer[i] = layer + 1
        wire_layer[j] = layer + 1
        depth = max(depth, layer + 1)
    return depth

def odd_even_transposition_network():
    """Generate the odd-even transposition sorting network."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def random_network(num_comparators):
    """Generate a random network with given number of comparators."""
    network = []
    for _ in range(num_comparators):
        network.append(random.choice(ALL_COMPARATORS))
    return network

def mutate(network, mutation_rate=0.1):
    """
    Mutate the network:
    - With some probability, swap two comparators
    - With some probability, insert a random comparator
    - With some probability, delete a random comparator
    - With some probability, change a comparator to a random one
    """
    new_network = list(network)
    n = len(new_network)
    
    if n == 0:
        # Start with a small random network
        for _ in range(random.randint(10, 30)):
            new_network.append(random.choice(ALL_COMPARATORS))
        return new_network
    
    # Decide mutation type
    r = random.random()
    if r < 0.3:
        # Swap two comparators
        if n >= 2:
            i, j = random.sample(range(n), 2)
            new_network[i], new_network[j] = new_network[j], new_network[i]
    elif r < 0.5:
        # Insert a random comparator at a random position
        pos = random.randint(0, n)
        new_network.insert(pos, random.choice(ALL_COMPARATORS))
    elif r < 0.7:
        # Delete a random comparator
        if n > 5:  # Don't let it get too small
            pos = random.randint(0, n - 1)
            new_network.pop(pos)
    elif r < 0.9:
        # Change a comparator to a random one
        pos = random.randint(0, n - 1)
        new_network[pos] = random.choice(ALL_COMPARATORS)
    else:
        # Local improvement: try to fix a specific issue
        # Pick a random position and try nearby comparators
        pos = random.randint(0, n - 1)
        old_c = new_network[pos]
        i, j = old_c
        # Try changing to a nearby comparator
        candidates = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < nj < N and (ni, nj) != old_c:
                    candidates.append((ni, nj))
        if candidates:
            new_network[pos] = random.choice(candidates)
    
    return new_network

def simulated_annealing():
    """
    Simulated annealing to find a small valid sorting network.
    """
    # Start with the odd-even transposition network (120 comparators)
    current_network = odd_even_transposition_network()
    current_valid = is_valid_network(current_network)
    best_network = list(current_network)
    best_valid = current_valid
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while T > T_min:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Generate a candidate
        candidate = mutate(current_network)
        
        # Quick check: if candidate is smaller than best, do full check
        if len(candidate) < len(best_network):
            if is_valid_network(candidate):
                best_network = list(candidate)
                best_valid = True
                # Restart temperature since we found improvement
                T = max(T * 0.9, T_min)
                current_network = list(candidate)
                current_valid = True
                continue
        
        # Partial check for fitness
        # Use a scoring function: number of failed inputs in a sample
        # For speed, we'll use a smaller sample
        sample_size = 512
        candidate_ok, _ = check_network_partial(candidate, sample_size)
        current_ok, _ = check_network_partial(current_network, 512) if not current_valid else (True, 0)
        
        if not current_valid:
            current_ok, _ = check_network_partial(current_network, 512)
        
        # If candidate is valid in sample and current is not, accept
        if candidate_ok and not current_ok:
            current_network = candidate
            current_valid = False  # Need full check later
            # Occasionally do full check
            if random.random() < 0.1:
                if is_valid_network(current_network):
                    current_valid = True
                    if len(current_network) < len(best_network):
                        best_network = list(current_network)
                        best_valid = True
        elif not candidate_ok and current_ok:
            # Reject
            pass
        else:
            # Both similar in sample, accept with some probability
            # Prefer smaller networks
            if len(candidate) < len(current_network):
                current_network = candidate
                current_valid = False
            elif random.random() < T:
                current_network = candidate
                current_valid = False
        
        # Cool down
        T *= cooling_rate
    
    # Final full check of best
    if not best_valid or not is_valid_network(best_network):
        # Fall back to odd-even transposition
        return odd_even_transposition_network()
    
    return best_network

def construct():
    """
    Construct a sorting network for 16 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    global DEADLINE
    DEADLINE = time.time() + MAX_TIME
    
    # Try simulated annealing
    try:
        network = simulated_annealing()
    except Exception:
        network = odd_even_transposition_network()
    
    # Final validation
    if not is_valid_network(network):
        network = odd_even_transposition_network()
    
    return network

# EVOLVE-BLOCK-START
# (The construct function is defined above, which is the entry point)
# EVOLVE-BLOCK-END
