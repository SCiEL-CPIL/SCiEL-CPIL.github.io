import numpy as np
import time
import random

N = 16

def _verify_network(comparators, n=16):
    """
    Verifies if the network sorts all binary inputs.
    Uses NumPy for speed.
    """
    inputs = np.array([list(map(int, format(i, f'0{n}b'))) for i in range(2**n)])
    for i, j in comparators:
        mask = inputs[:, i] > inputs[:, j]
        if np.any(mask):
            inputs[mask, i] = 1 - inputs[mask, i]
            inputs[mask, j] = 1 - inputs[mask, j]
    return np.array_equal(inputs, np.sort(inputs, axis=1))

def _batcher_oes(n):
    """
    Generates Batcher's Odd-Even Merge Sort network.
    Returns a list of (i, j) pairs.
    """
    if n == 1:
        return []
    
    def merge(n, offset=0, step=2):
        if n == 1:
            return []
        half = n // 2
        
        odd_comps = merge(half, offset + step, 2 * step)
        even_comps = merge(half, offset, 2 * step)
        
        final_comps = []
        for k in range(half):
            i = offset + 2 * k * step
            j = offset + (2 * k + 1) * step
            final_comps.append((i, j))
        
        return odd_comps + even_comps + final_comps
    
    half = n // 2
    
    even_sort = _batcher_oes(half)
    even_sort_remap = [(2 * i, 2 * j) for (i, j) in even_sort]
    
    odd_sort = _batcher_oes(half)
    odd_sort_remap = [(2 * i + 1, 2 * j + 1) for (i, j) in odd_sort]
    
    merge_comps = merge(n, 0, 1)
    
    return even_sort_remap + odd_sort_remap + merge_comps

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute all 2^16 binary inputs and their sorted versions
    all_inputs = np.array([list(map(int, format(i, f'0{N}b'))) for i in range(2**N)])
    sorted_inputs = np.sort(all_inputs, axis=1)
    
    def verify_fast(comparators):
        state = all_inputs.copy()
        for i, j in comparators:
            mask = state[:, i] > state[:, j]
            if np.any(mask):
                state[mask, i] = 1 - state[mask, i]
                state[mask, j] = 1 - state[mask, j]
        return np.array_equal(state, sorted_inputs)
    
    def count_errors(comparators):
        """Return number of inputs that are not sorted."""
        state = all_inputs.copy()
        for i, j in comparators:
            mask = state[:, i] > state[:, j]
            if np.any(mask):
                state[mask, i] = 1 - state[mask, i]
                state[mask, j] = 1 - state[mask, j]
        return np.sum(~np.array_equal(state, sorted_inputs, axis=1))
    
    # Generate Batcher's OES network as starting point
    network = _batcher_oes(N)
    
    if not verify_fast(network):
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    current_network = network[:]
    current_errors = count_errors(current_network)
    
    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    best_network = current_network[:]
    best_errors = current_errors
    
    while time.time() < deadline - 3.0:
        if current_errors == 0:
            best_network = current_network[:]
            best_errors = 0
            # Try to remove comparators
            for k in range(len(current_network) - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                trial = current_network[:k] + current_network[k+1:]
                if verify_fast(trial):
                    current_network = trial
                    current_errors = 0
                    best_network = current_network[:]
                    best_errors = 0
                else:
                    current_errors = count_errors(current_network)
        else:
            # Perform a random move
            move_type = random.random()
            
            if move_type < 0.5 and len(current_network) > 10:
                # Remove a random comparator
                k = random.randint(0, len(current_network) - 1)
                trial = current_network[:k] + current_network[k+1:]
                trial_errors = count_errors(trial)
                if trial_errors <= current_errors or random.random() < np.exp(-(trial_errors - current_errors) / T):
                    current_network = trial
                    current_errors = trial_errors
                    if current_errors < best_errors:
                        best_network = current_network[:]
                        best_errors = current_errors
            elif move_type < 0.8:
                # Swap two comparators
                if len(current_network) >= 2:
                    i, j = random.sample(range(len(current_network)), 2)
                    trial = current_network[:]
                    trial[i], trial[j] = trial[j], trial[i]
                    trial_errors = count_errors(trial)
                    if trial_errors <= current_errors or random.random() < np.exp(-(trial_errors - current_errors) / T):
                        current_network = trial
                        current_errors = trial_errors
                        if current_errors < best_errors:
                            best_network = current_network[:]
                            best_errors = current_errors
            else:
                # Insert a random comparator
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                k = random.randint(0, len(current_network))
                trial = current_network[:k] + [(i, j)] + current_network[k:]
                trial_errors = count_errors(trial)
                if trial_errors <= current_errors or random.random() < np.exp(-(trial_errors - current_errors) / T):
                    current_network = trial
                    current_errors = trial_errors
                    if current_errors < best_errors:
                        best_network = current_network[:]
                        best_errors = current_errors
        
        T *= cooling
    
    # Final greedy peeling on best network
    if best_errors == 0:
        current_network = best_network[:]
        for _pass in range(2):
            if time.time() > deadline - 1.0:
                break
            for k in range(len(current_network) - 1, -1, -1):
                if time.time() > deadline - 0.5:
                    break
                trial = current_network[:k] + current_network[k+1:]
                if verify_fast(trial):
                    current_network = trial
    
    return current_network
# EVOLVE-BLOCK-END
