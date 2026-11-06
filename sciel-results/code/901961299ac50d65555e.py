import time
import random

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using Bit-Vector Simulation (Python Integers)
    # State: List of N Python integers. Each integer has NUM_INPUTS bits.
    # Bit k of state[i] represents the value of wire i for the k-th input.
    
    # Pre-compute initial state for all inputs
    # Input k is the integer k. Bit j of input k is (k >> j) & 1.
    # We want state[j] to be an integer where bit k is 1 if input k has 1 on wire j.
    # So state[j] = sum( (1 << k) for k in range(NUM_INPUTS) if (k >> j) & 1 )
    
    # Efficient way to generate these masks:
    # We can build them iteratively or use a pattern.
    # For wire j, the pattern of bits is 0...01...10...1 repeated.
    # Period is 2^(j+1). Low 2^j bits are 0, next 2^j bits are 1.
    
    initial_state = [0] * N
    for j in range(N):
        mask = 0
        # Build the mask for wire j
        # Instead of looping 131072 times, we can use bit manipulation tricks or just loop once.
        # 131072 iterations is fast enough (< 10ms).
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        initial_state[j] = mask

    def is_valid(network):
        # Copy state
        state = initial_state[:]
        
        for i, j in network:
            # min to i, max to j
            # min_val = state[i] & state[j]
            # max_val = state[i] | state[j]
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        # Check if sorted: for all k, state[k] <= state[k+1] bitwise
        # This means if state[k+1] has a 0, state[k] must have a 0.
        # i.e., state[k] & ~state[k+1] must be 0.
        for k in range(N - 1):
            if (state[k] & ~state[k+1]) != 0:
                return False
        return True

    # Initial check (should be valid)
    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    # 3. Local Search with Random Perturbations
    # Strategy:
    # 1. Greedy removal: Try to remove each comparator. If valid, remove it and restart scan.
    # 2. Perturbation: If no removal is possible, randomly mutate the network (change a comparator)
    #    and accept if valid. This explores the space of valid networks of the same size.
    #    This allows us to find structures that *allow* removals.
    
    while time.time() < deadline:
        improved = False
        found_removal = False
        
        # Attempt removals
        for idx in range(len(best_network)):
            if time.time() > deadline:
                break
            
            # Create candidate by removing idx
            candidate = best_network[:idx] + best_network[idx+1:]
            
            if is_valid(candidate):
                best_network = candidate
                improved = True
                found_removal = True
                break # Restart scan from beginning
        
        if not found_removal:
            # Perturb: Replace a random comparator with a new random one
            if len(best_network) > 0:
                # Try a few random mutations before giving up on this "stuck" state
                # to avoid getting stuck in a local minimum of the mutation space
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    
                    idx = random.randint(0, len(best_network) - 1)
                    
                    # Generate a new random comparator
                    new_i = random.randint(0, N - 1)
                    new_j = random.randint(0, N - 1)
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    if new_i == new_j:
                        continue
                    
                    candidate = best_network[:]
                    candidate[idx] = (new_i, new_j)
                    
                    if is_valid(candidate):
                        best_network = candidate
                        improved = True
                        break # Restart removal scan

    return best_network
# EVOLVE-BLOCK-END
