import time
import random
import math
import numpy as np

N = 17
NUM_INPUTS = 1 << N

# Pre-compute target states for the zero-one principle
# For each input k (0 to 2^N - 1), the sorted output has 1s at the end.
# Let w = popcount(k). The output has 1s at indices N-w, ..., N-1.
# So for wire i, the value is 1 if i >= N - w, else 0.
# We build an integer mask for each wire i, where bit k is the value of wire i for input k.
TARGET_STATES = [0] * N
for i in range(N):
    mask = 0
    for k in range(NUM_INPUTS):
        # Calculate number of 1s in k
        w = bin(k).count('1')
        # Check if wire i should be 1 for this input
        if i >= N - w:
            mask |= (1 << k)
    TARGET_STATES[i] = mask

def _generate_seed_network():
    """Generate the odd-even transposition network."""
    net = []
    for phase in range(N):
        # Even phase: (0,1), (2,3), ...
        # Odd phase: (1,2), (3,4), ...
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _verify_network(network):
    """
    Verify if the network sorts all binary inputs.
    Uses bit-vector simulation.
    state[i] is an integer where bit k represents the value of wire i for input k.
    """
    # Initialize state with input wires
    # Wire j for input k is bit j of k.
    # So state[j] has bit k set if (k >> j) & 1 is 1.
    state = [0] * N
    for j in range(N):
        mask = 0
        for k in range(NUM_INPUTS):
            if (k >> j) & 1:
                mask |= (1 << k)
        state[j] = mask
        
    # Apply comparators
    for i, j in network:
        # min to i, max to j
        # new_i = state[i] & state[j]  (1 only if both 1)
        # new_j = state[i] | state[j]  (1 if at least one 1)
        si = state[i]
        sj = state[j]
        state[i] = si & sj
        state[j] = si | sj
        
    # Check against target states
    for i in range(N):
        if state[i] != TARGET_STATES[i]:
            return False
    return True

def _try_reduce(network, max_rounds=2):
    """
    Greedy reduction: try to remove any comparator that is redundant.
    """
    current = network[:]
    removed_any = True
    rounds = 0
    while removed_any and rounds < max_rounds:
        removed_any = False
        rounds += 1
        # Shuffle to break symmetry
        indices = list(range(len(current)))
        random.shuffle(indices)
        
        for idx in indices:
            # Try removing current[idx]
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate):
                current = candidate
                removed_any = True
                break # Restart the loop with the new shorter list
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # 1. Initialize with seed network
    current_net = _generate_seed_network()
    best_net = current_net[:]
    
    # Initial reduction to get a good starting point
    current_net = _try_reduce(current_net, max_rounds=3)
    best_net = current_net[:]
    
    # 2. Simulated Annealing
    temperature = 10.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    # Pre-compute some random pairs for mutation
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline and temperature > min_temp:
        # Generate neighbor
        neighbor = current_net[:]
        n = len(neighbor)
        
        strategy = random.random()
        
        if strategy < 0.3:
            # Remove a comparator
            if n > 0:
                idx = random.randint(0, n - 1)
                neighbor.pop(idx)
                
        elif strategy < 0.5:
            # Add a comparator
            # Bias towards adding near existing comparators or random
            if random.random() < 0.5 and n > 0:
                # Add near a random existing comparator
                ref_idx = random.randint(0, n - 1)
                ref_i, ref_j = neighbor[ref_idx]
                # Perturb wires
                new_i = max(0, min(N - 2, ref_i + random.randint(-2, 2)))
                new_j = max(new_i + 1, min(N - 1, ref_j + random.randint(-2, 2)))
                if new_i < new_j:
                    pos = max(0, min(n, ref_idx + random.randint(-2, 2)))
                    neighbor.insert(pos, (new_i, new_j))
            else:
                # Random pair
                new_i, new_j = random.choice(all_pairs)
                pos = random.randint(0, n)
                neighbor.insert(pos, (new_i, new_j))
                
        elif strategy < 0.7:
            # Move a comparator
            if n > 1:
                idx = random.randint(0, n - 1)
                comp = neighbor.pop(idx)
                new_pos = random.randint(0, n - 1)
                neighbor.insert(new_pos, comp)
                
        elif strategy < 0.85:
            # Swap two comparators
            if n > 1:
                i, j = random.sample(range(n), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
        else:
            # Replace a comparator with a different one
            if n > 0:
                idx = random.randint(0, n - 1)
                old_i, old_j = neighbor[idx]
                new_i = max(0, min(N - 2, old_i + random.randint(-3, 3)))
                new_j = max(new_i + 1, min(N - 1, old_j + random.randint(-3, 3)))
                if new_i < new_j:
                    neighbor[idx] = (new_i, new_j)

        # Verify
        if not _verify_network(neighbor):
            temperature *= cooling_rate
            continue
        
        # Try to reduce the neighbor immediately
        neighbor = _try_reduce(neighbor, max_rounds=1)
        
        # Acceptance
        curr_cost = len(current_net)
        new_cost = len(neighbor)
        delta = new_cost - curr_cost
        
        if delta <= 0:
            current_net = neighbor
            if new_cost < len(best_net):
                best_net = current_net[:]
        else:
            # Accept with probability exp(-delta/T)
            if temperature > 0:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_net = neighbor
        
        temperature *= cooling_rate

    # Final polish
    if time.time() < deadline:
        final_net = _try_reduce(current_net, max_rounds=5)
        if len(final_net) < len(best_net):
            best_net = final_net
    
    return best_net

