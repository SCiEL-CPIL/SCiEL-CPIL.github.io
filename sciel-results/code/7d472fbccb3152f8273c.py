import numpy as np
import time
import random

N = 18

# Pre-compute all boolean inputs for verification
# Shape: (2^N, N)
# inputs[k, j] is the j-th bit of the integer k
ALL_INPUTS = None

def get_all_inputs():
    global ALL_INPUTS
    if ALL_INPUTS is None:
        n_inputs = 1 << N
        indices = np.arange(n_inputs, dtype=np.uint16)
        ALL_INPUTS = np.zeros((n_inputs, N), dtype=np.uint8)
        for j in range(N):
            ALL_INPUTS[:, j] = (indices >> j) & 1
    return ALL_INPUTS

def verify_network(net):
    """
    Returns True if the network sorts all 2^N boolean inputs.
    Sorted means: w0 <= w1 <= ... <= wN-1
    """
    if not net:
        return N <= 1
    
    # Copy the inputs because we modify them
    inputs = get_all_inputs().copy()
    
    for i, j in net:
        min_ij = np.minimum(inputs[:, i], inputs[:, j])
        max_ij = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_ij
        inputs[:, j] = max_ij
        
    # Check if sorted in ascending order
    return np.all(inputs[:, :-1] <= inputs[:, 1:])

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    # 1. Generate initial network using Odd-Even Transposition Sort
    def generate_oet_network(n):
        comps = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comps.append((i, i + 1))
        return comps

    best_net = generate_oet_network(N)
    best_count = len(best_net)
    
    # Verify seed
    if not verify_network(best_net):
        # Fallback to a simple bubble sort network if OET fails (unlikely)
        best_net = [(i, i+1) for phase in range(N) for i in range(phase % 2, N-1, 2)]
        best_count = len(best_net)

    current_net = list(best_net)
    current_count = best_count
    
    # Pre-compute all valid pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Mutation functions
    def mutate_remove(net):
        if not net:
            return net
        idx = random.randint(0, len(net)-1)
        return net[:idx] + net[idx+1:]

    def mutate_add(net):
        i, j = random.choice(all_pairs)
        idx = random.randint(0, len(net))
        return net[:idx] + [(i, j)] + net[idx:]

    def mutate_swap(net):
        if len(net) < 2:
            return net
        i, j = random.sample(range(len(net)), 2)
        new_net = net[:]
        new_net[i], new_net[j] = new_net[j], new_net[i]
        return new_net

    def mutate_modify(net):
        if not net:
            return net
        idx = random.randint(0, len(net)-1)
        new_i, new_j = random.choice(all_pairs)
        new_net = net[:]
        new_net[idx] = (new_i, new_j)
        return new_net

    mutations = [mutate_remove, mutate_add, mutate_swap, mutate_modify]
    
    # Search Loop
    while time.time() < deadline:
        # Choose a mutation
        # Bias towards removal to minimize count
        if current_count > 80:
            # More likely to remove
            mut = random.choice([mutate_remove, mutate_remove, mutate_remove, mutate_add, mutate_swap, mutate_modify])
        elif current_count > 60:
            mut = random.choice([mutate_remove, mutate_remove, mutate_add, mutate_swap, mutate_modify])
        else:
            # Near optimal, try all
            mut = random.choice(mutations)
            
        new_net = mut(current_net)
        new_count = len(new_net)
        
        # We only accept if count is not increasing
        # (Adding comparators is only useful if it enables future removals, 
        # but simple hill climbing doesn't track that. 
        # However, accepting equal count allows rearrangement.)
        if new_count > current_count:
            continue
            
        # Verify
        if verify_network(new_net):
            # Accept
            current_net = new_net
            current_count = new_count
            
            if new_count < best_count:
                best_net = list(current_net)
                best_count = new_count
                
    return best_net

# EVOLVE-BLOCK-END
