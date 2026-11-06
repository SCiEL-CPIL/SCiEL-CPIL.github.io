import numpy as np
import time
import random

N = 18
NUM_INPUTS = 1 << N  # 262144

def _make_all_inputs():
    return np.arange(NUM_INPUTS, dtype=np.uint32)

def _apply_comparator(inputs, i, j):
    bit_i = (inputs >> i) & 1
    bit_j = (inputs >> j) & 1
    new_i = np.minimum(bit_i, bit_j)
    new_j = np.maximum(bit_i, bit_j)
    inputs = inputs & ~((1 << i) | (1 << j))
    inputs = inputs | (new_i << i) | (new_j << j)
    return inputs

def _is_sorted_batch(inputs):
    result = np.ones(len(inputs), dtype=bool)
    for k in range(N - 1):
        bit_k = (inputs >> k) & 1
        bit_k1 = (inputs >> (k + 1)) & 1
        result &= ~(bit_k & ~bit_k1)
    return result

def _verify_network(network):
    inputs = _make_all_inputs()
    for (i, j) in network:
        inputs = _apply_comparator(inputs, i, j)
    return _is_sorted_batch(inputs).all()

def _odd_even_transposition():
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def construct():
    start_time = time.time()
    deadline = start_time + 55
    random.seed(42)
    
    network = _odd_even_transposition()
    best_network = list(network)
    best_count = len(network)
    
    if not _verify_network(network):
        return best_network
    
    # Phase 1: Greedy removal with random ordering
    while time.time() < deadline:
        removed = False
        positions = list(range(len(network)))
        random.shuffle(positions)
        for idx in positions:
            if time.time() > deadline:
                break
            trial = network[:idx] + network[idx+1:]
            if _verify_network(trial):
                network = trial
                best_network = list(network)
                best_count = len(network)
                removed = True
                break
        if not removed:
            break
    
    # Phase 2: Swap-then-remove to escape local optima
    while time.time() < deadline:
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            if len(network) < 2:
                break
            idx = random.randint(0, len(network) - 2)
            trial = list(network)
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            for remove_idx in [idx, idx+1]:
                trial2 = trial[:remove_idx] + trial[remove_idx+1:]
                if _verify_network(trial2):
                    network = trial2
                    best_network = list(network)
                    best_count = len(network)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_network
