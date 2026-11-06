import numpy as np
import time
import random

N = 16
NUM_INPUTS = 1 << N

def _generate_batcher_oes(n):
    """
    Generates Batcher's Odd-Even Merge Sort network.
    """
    if n == 1:
        return []
    
    def merge(n, offset=0, step=2):
        if n == 1:
            return []
        half = n // 2
        
        # Merge odd-indexed subsequence
        odd_comps = merge(half, offset + step, 2 * step)
        # Merge even-indexed subsequence
        even_comps = merge(half, offset, 2 * step)
        
        # Final stage: compare adjacent pairs in the merged sequence
        final_comps = []
        for k in range(half):
            i = offset + 2 * k * step
            j = offset + (2 * k + 1) * step
            final_comps.append((i, j))
        
        return odd_comps + even_comps + final_comps

    half = n // 2
    
    # S(half) on even indices
    even_sort = _generate_batcher_oes(half)
    even_sort_remap = [(2 * i, 2 * j) for (i, j) in even_sort]
    
    # S(half) on odd indices
    odd_sort = _generate_batcher_oes(half)
    odd_sort_remap = [(2 * i + 1, 2 * j + 1) for (i, j) in odd_sort]
    
    # Merge(n)
    merge_comps = merge(n, 0, 1)
    
    return even_sort_remap + odd_sort_remap + merge_comps

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute inputs
    inputs = np.array([list(map(int, format(i, f'0{N}b'))) for i in range(NUM_INPUTS)])
    
    def apply_network(comparators, initial_state):
        state = initial_state.copy()
        for i, j in comparators:
            mask = state[:, i] > state[:, j]
            if np.any(mask):
                tmp = state[mask, i].copy()
                state[mask, i] = state[mask, j]
                state[mask, j] = tmp
        return state

    def check_validity(state):
        diff = state[:, 1:] - state[:, :-1]
        return np.sum(diff < 0)

    # Initial network
    current_network = _generate_batcher_oes(N)
    current_state = apply_network(current_network, inputs)
    current_violations = check_validity(current_state)
    current_count = len(current_network)
    
    best_network = current_network[:]
    best_count = current_count
    best_violations = current_violations
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.998
    
    while time.time() < deadline - 2.0:
        if T < T_min:
            T = T_min
            
        move_type = random.random()
        new_network = current_network[:]
        
        if move_type < 0.7:
            # Delete
            if current_count > 0:
                idx = random.randint(0, current_count - 1)
                new_network.pop(idx)
                new_count = current_count - 1
            else:
                new_count = 0
        elif move_type < 0.85:
            # Insert
            i = random.randint(0, N - 1)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, current_count)
            new_network.insert(pos, (i, j))
            new_count = current_count + 1
        elif move_type < 0.95:
            # Swap
            if current_count > 1:
                idx1 = random.randint(0, current_count - 1)
                idx2 = random.randint(0, current_count - 1)
                if idx1 != idx2:
                    new_network[idx1], new_network[idx2] = new_network[idx2], new_network[idx1]
            new_count = current_count
        else:
            # Mutate
            if current_count > 0:
                idx = random.randint(0, current_count - 1)
                i, j = new_network[idx]
                if random.random() < 0.5:
                    new_i = random.randint(0, N - 1)
                    if new_i < j:
                        new_network[idx] = (new_i, j)
                    elif new_i > j:
                        new_network[idx] = (j, new_i)
                else:
                    new_j = random.randint(0, N - 1)
                    if new_j > i:
                        new_network[idx] = (i, new_j)
                    elif new_j < i:
                        new_network[idx] = (new_j, i)
            new_count = current_count

        new_state = apply_network(new_network, inputs)
        new_violations = check_validity(new_state)
        
        current_energy = current_count + 100 * current_violations
        new_energy = new_count + 100 * new_violations
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or (T > 0 and random.random() < np.exp(-delta_e / T)):
            current_network = new_network
            current_state = new_state
            current_count = new_count
            current_violations = new_violations
            
            if current_violations == 0 and current_count < best_count:
                best_count = current_count
                best_network = current_network[:]
                best_violations = 0
            elif best_violations > 0 and current_violations < best_violations:
                # Keep track of best invalid to guide search?
                # Not strictly necessary if we have a valid best.
                pass

        T *= cooling_rate

    # Final check
    if best_violations == 0:
        return best_network
    else:
        # Return initial valid network
        return _generate_batcher_oes(N)

# EVOLVE-BLOCK-END
