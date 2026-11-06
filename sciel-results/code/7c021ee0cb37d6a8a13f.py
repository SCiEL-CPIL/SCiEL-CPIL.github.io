import numpy as np
import time
import random
import math

N = 17
NUM_INPUTS = 1 << N
NUM_WORDS = NUM_INPUTS // 64

PACKED_INPUTS = None

def construct():
    global PACKED_INPUTS
    if PACKED_INPUTS is None:
        indices = np.arange(NUM_INPUTS, dtype=np.uint32)
        bits = (indices[:, None] >> np.arange(N)[None, :]) & 1
        bits_reshaped = bits.reshape(NUM_WORDS, 64, N).transpose(0, 2, 1)
        powers = 1 << np.arange(64, dtype=np.uint64)
        PACKED_INPUTS = (bits_reshaped * powers[None, None, :]).sum(axis=2).astype(np.uint64)

    start_time = time.time()
    deadline = start_time + 55.0

    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))

    def is_valid(network):
        if not network:
            return N == 1
        
        state = PACKED_INPUTS.copy()
        
        for i, j in network:
            min_val = state[:, i] & state[:, j]
            max_val = state[:, i] | state[:, j]
            state[:, i] = min_val
            state[:, j] = max_val

        for k in range(N - 1):
            if np.any(state[:, k] & ~state[:, k+1]):
                return False
        return True

    def reduce_network(network):
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for idx in range(len(network)):
                if time.time() > deadline:
                    return network
                candidate = network[:idx] + network[idx+1:]
                if is_valid(candidate):
                    network = candidate
                    changed = True
                    break
        return network

    best_network = reduce_network(seed_network[:])
    best_cost = len(best_network)
    
    current_network = best_network[:]
    current_cost = best_cost
    
    random.seed(42)
    
    T = 10.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        if T < 0.01:
            T = 10.0
            
        mutation_type = random.random()
        candidate = None
        
        if mutation_type < 0.6:
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j: i, j = j, i
                candidate = current_network[:]
                pos = random.randint(0, len(candidate))
                candidate.insert(pos, (i, j))
        elif mutation_type < 0.8:
            if len(current_network) > 0:
                candidate = current_network[:]
                idx = random.randint(0, len(candidate) - 1)
                candidate.pop(idx)
        else:
            if len(current_network) > 0:
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i != j:
                    if i > j: i, j = j, i
                    candidate = current_network[:]
                    idx = random.randint(0, len(candidate) - 1)
                    candidate[idx] = (i, j)
        
        if candidate is None:
            continue

        if is_valid(candidate):
            optimized = reduce_network(candidate)
            new_cost = len(optimized)
            
            if new_cost < current_cost:
                current_network = optimized
                current_cost = new_cost
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    current_network = optimized
                    current_cost = new_cost
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_network = optimized
                        current_cost = new_cost
        
        T *= cooling_rate
        
    return best_network
