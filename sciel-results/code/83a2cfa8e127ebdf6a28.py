import time
import random
import math
import numpy as np

N = 17
NUM_INPUTS = 1 << N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit

    # 1. Generate the seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = seed_network[:]
    
    # 2. Verification Function using Bit-Sliced NumPy Simulation
    # State is a (NUM_INPUTS, N) array of uint64? No, that's 131k * 17 * 8 bytes ~ 17MB.
    # Better: (N, NUM_INPUTS/64) array of uint64.
    # 131072 / 64 = 2048 words.
    # N=17, so 17 * 2048 * 8 bytes = 286 KB. Very fast to copy and operate on.
    
    NUM_WORDS = NUM_INPUTS // 64
    
    def initial_state():
        state = np.zeros((N, NUM_WORDS), dtype=np.uint64)
        for j in range(N):
            # We need to set bit k in word w if the k-th input has a 1 at position j.
            # Input k is a number 0..131071. Bit j of k is 1 if (k >> j) & 1.
            # Word w contains inputs 64*w .. 64*w+63.
            # For a fixed j, the pattern of bits in the inputs repeats every 2^(j+1).
            # Actually, it's easier to just construct the masks for the first 64 inputs and repeat?
            # No, the inputs are 0, 1, 2, ...
            # Let's generate the masks for the first 64 inputs and broadcast?
            # For word w, inputs are I = 64*w + i for i in 0..63.
            # Bit j of I is 1 if (64*w + i) has bit j set.
            # If j < 6, bit j depends only on i.
            # If j >= 6, bit j depends on w and i.
            
            # Let's just compute it simply for the first 64 and then handle the shift for w.
            # Actually, for j in 0..16:
            # The bit j of the input number K is 1 for half the inputs.
            # In a block of 64 consecutive inputs, the pattern of bit j is:
            # If j < 6: The pattern is periodic with period 2^(j+1).
            #   Example j=0: 0,1,0,1...
            #   Example j=1: 0,0,1,1...
            # If j >= 6: The bit is constant for the whole block of 64, determined by w.
            
            # Let's precompute the mask for the low 6 bits (i part) for each j.
            # And for the high bits, it's just a constant mask (all 1s or all 0s) for the word.
            
            mask_low = 0
            for i in range(64):
                if (i >> j) & 1:
                    mask_low |= (1 << i)
            
            for w in range(NUM_WORDS):
                # The input range is [64*w, 64*w + 63]
                # Bit j of (64*w + i) is ( (64*w) >> j | (i >> j) ) & 1 ? No.
                # (A+B) >> j is not (A>>j) + (B>>j) if there is carry.
                # However, 64*w is a multiple of 64.
                # If j < 6: 64*w has 0 in bit j. So bit j of sum is bit j of i.
                # If j >= 6: i < 64, so i has 0 in bit j. Bit j of sum is bit (j-6) of w.
                
                if j < 6:
                    state[j, w] = mask_low
                else:
                    if (w >> (j - 6)) & 1:
                        state[j, w] = (1 << 64) - 1 # All 1s
                    else:
                        state[j, w] = 0
        
        return state

    base_state = initial_state()

    def is_valid(network):
        # Copy base state
        state = base_state.copy()
        
        for i, j in network:
            # min(state[i], state[j]) = state[i] & state[j]
            # max(state[i], state[j]) = state[i] | state[j]
            # Vectorized bitwise ops
            new_i = state[i] & state[j]
            new_j = state[i] | state[j]
            state[i] = new_i
            state[j] = new_j
        
        # Check sorted: state[k] <= state[k+1] for all k
        # This means state[k] & ~state[k+1] == 0
        # If any bit is set where state[k] is 1 and state[k+1] is 0, it's unsorted.
        
        for k in range(N - 1):
            # Check if there is any input where wire k is 1 and wire k+1 is 0
            violation = state[k] & ~state[k+1]
            if np.any(violation != 0):
                return False
        
        return True

    if not is_valid(best_network):
        return best_network

    random.seed(42)
    
    # 3. Aggressive removal function
    def try_remove_all(current_net, time_limit=None):
        removed = True
        while removed:
            if time.time() > time_limit:
                break
            removed = False
            n = len(current_net)
            indices = list(range(n))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > time_limit:
                    break
                candidate = current_net[:idx] + current_net[idx+1:]
                if is_valid(candidate):
                    current_net = candidate
                    removed = True
                    break
        return current_network

    # Initial aggressive cleanup
    current_network = try_remove_all(best_network, time_limit=deadline)
    if len(current_network) < len(best_network):
        best_network = current_network[:]

    # Simulated Annealing parameters
    initial_temp = 5.0
    cooling_rate = 0.999
    min_temp = 0.05
    
    temperature = initial_temp
    
    # Precompute some useful data for mutations?
    # Maybe just stick to the existing strategies but with faster validation.
    
    while time.time() < deadline and temperature > min_temp:
        if time.time() > deadline:
            break
        
        # Generate a neighbor
        neighbor = current_network[:]
        strategy = random.random()
        n = len(neighbor)
        
        if strategy < 0.20:
            # Add a random comparator - biased toward useful positions
            # Bias towards adjacent wires or wires that are far apart?
            # Let's try random first, but maybe bias towards (i, i+1) or (i, i+2)
            r = random.random()
            if r < 0.5:
                new_i = random.randint(0, N - 2)
                new_j = new_i + 1
            elif r < 0.8:
                new_i = random.randint(0, N - 3)
                new_j = new_i + 2
            else:
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i > new_j:
                    new_i, new_j = new_j, new_i
                if new_i == new_j:
                    new_i = (new_i + 1) % N
                    new_j = new_i
            
            if new_i != new_j:
                # Insert at a random position
                pos = random.randint(0, len(neighbor))
                neighbor.insert(pos, (new_i, new_j))
        
        elif strategy < 0.40:
            # Remove a random comparator
            if n > 1:
                idx = random.randint(0, n - 1)
                neighbor.pop(idx)
        
        elif strategy < 0.55:
            # Move a comparator to a different position
            if n > 2:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    comp = neighbor.pop(i)
                    if i < j:
                        j -= 1
                    neighbor.insert(j, comp)
        
        elif strategy < 0.70:
            # Swap two comparators
            if n > 1:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        elif strategy < 0.85:
            # Replace a comparator with a nearby wire pair
            if n > 0:
                i = random.randint(0, n - 1)
                old_i, old_j = neighbor[i]
                new_i = max(0, min(N - 2, old_i + random.randint(-2, 2)))
                new_j = max(new_i + 1, min(N - 1, old_j + random.randint(-2, 2)))
                if new_i < new_j:
                    neighbor[i] = (new_i, new_j)
        
        else:
            # Reorder a segment of the network
            if n > 3:
                start = random.randint(0, n - 4)
                end = random.randint(start + 2, n - 1)
                segment = neighbor[start:end+1]
                random.shuffle(segment)
                neighbor[start:end+1] = segment
        
        # Check validity
        if not is_valid(neighbor):
            temperature *= cooling_rate
            continue
        
        # Acceptance decision
        current_cost = len(current_network)
        neighbor_cost = len(neighbor)
        delta = neighbor_cost - current_cost
        
        if delta <= 0:
            # Accept improvement
            current_network = neighbor
            if len(current_network) < len(best_network):
                best_network = current_network[:]
        else:
            # Accept worse solution with probability exp(-delta/T)
            if temperature > 0:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_network = neighbor
        
        temperature *= cooling_rate
    
    # Final cleanup with aggressive removal
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            final_network = try_remove_all(current_network, time_limit=deadline)
            if len(final_network) < len(best_network):
                best_network = final_network[:]
                current_network = best_network[:]
    
    return best_network
# EVOLVE-BLOCK-END
