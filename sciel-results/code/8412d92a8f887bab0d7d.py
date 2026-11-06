"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
import random

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N wires.
    Returns a list of (i, j) comparators.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            return step1 + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def _apply_network_vectorized(comps, inputs):
    """
    Apply the network to a batch of inputs (numpy array of shape (batch_size, N)).
    """
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _check_network_fast(comps, bits):
    """
    Check if the network sorts all boolean inputs. Returns (is_valid, num_failures).
    'bits' is a numpy array of shape (2^N, N).
    """
    vec = bits.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    
    diffs = np.diff(vec, axis=1)
    failures = np.any(diffs < 0, axis=1)
    return (not np.any(failures)), np.sum(failures)

def _verify_network(comparators, n=N):
    """Verify network sorts all 2^n Boolean inputs using zero-one principle."""
    for mask in range(1 << n):
        wires = [(mask >> i) & 1 for i in range(n)]
        for i, j in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        if wires != sorted(wires):
            return False
    return True

def _generate_random_comparator():
    """Generate a random valid comparator (i, j) with i < j."""
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    return (i, j)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s safety margin for 90s limit
    
    # Pre-compute all boolean inputs for vectorized checking
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # Generate initial network (Batcher's, 63 comparators for N=16)
    current_comps = _generate_batcher_network()
    
    # Verify initial network (sanity check)
    is_valid, _ = _check_network_fast(current_comps, bits)
    if not is_valid:
        # Fallback to a simple valid network
        current_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    current_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    current_comps.append((j, j + 1))
        if not _verify_network(current_comps):
            return current_comps

    best_comps = list(current_comps)
    best_count = len(current_comps)
    
    # Simulated annealing parameters
    initial_temp = 1.0
    cooling_rate = 0.9995
    min_temp = 1e-6
    max_comparators = 100  # Don't let network grow too large
    
    # Precompute all possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    temp = initial_temp
    step_count = 0
    
    while time.time() < deadline and temp > min_temp:
        step_count += 1
        
        # Choose a move type with probabilities
        move_type = random.random()
        candidate_comps = list(current_comps)
        
        if move_type < 0.4 and len(candidate_comps) > 0:
            # Move 1: Remove a random comparator
            idx = random.randint(0, len(candidate_comps) - 1)
            candidate_comps.pop(idx)
            
        elif move_type < 0.6 and len(candidate_comps) < max_comparators:
            # Move 2: Insert a random comparator at a random position
            new_comp = _generate_random_comparator()
            pos = random.randint(0, len(candidate_comps))
            candidate_comps.insert(pos, new_comp)
            
        elif move_type < 0.8 and len(candidate_comps) >= 2:
            # Move 3: Swap two adjacent comparators
            idx = random.randint(0, len(candidate_comps) - 2)
            candidate_comps[idx], candidate_comps[idx + 1] = candidate_comps[idx + 1], candidate_comps[idx]
            
        else:
            # Move 4: Relocate a comparator to a different position
            if len(candidate_comps) >= 2:
                idx = random.randint(0, len(candidate_comps) - 1)
                comp = candidate_comps.pop(idx)
                new_pos = random.randint(0, len(candidate_comps))
                candidate_comps.insert(new_pos, comp)
        
        # Check if candidate is valid
        is_valid, failures = _check_network_fast(candidate_comps, bits)
        
        # Compute energy: we want to minimize comparator count, but must be valid
        # If valid, energy = number of comparators
        # If invalid, energy = number of comparators + penalty * failures
        if is_valid:
            candidate_energy = len(candidate_comps)
        else:
            candidate_energy = len(candidate_comps) + 10.0 * failures
        
        current_is_valid, _ = _check_network_fast(current_comps, bits)
        if current_is_valid:
            current_energy = len(current_comps)
        else:
            _, cur_failures = _check_network_fast(current_comps, bits)
            current_energy = len(current_comps) + 10.0 * cur_failures
        
        delta_energy = candidate_energy - current_energy
        
        # Acceptance decision
        if delta_energy < 0:
            # Always accept improvement
            current_comps = candidate_comps
            
            if is_valid and len(current_comps) < best_count:
                best_comps = list(current_comps)
                best_count = len(current_comps)
        else:
            # Accept with probability exp(-delta_energy / temp)
            if temp > 0:
                accept_prob = np.exp(-delta_energy / temp)
                if random.random() < accept_prob:
                    current_comps = candidate_comps
        
        # Cool down
        temp *= cooling_rate
        
        # Periodic greedy removal pass to quickly reduce count
        if step_count % 500 == 0 and time.time() < deadline:
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for idx in range(len(current_comps) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    candidate = current_comps[:idx] + current_comps[idx+1:]
                    is_valid, _ = _check_network_fast(candidate, bits)
                    if is_valid:
                        current_comps = candidate
                        improved = True
                        if len(current_comps) < best_count:
                            best_comps = list(current_comps)
                            best_count = len(current_comps)
                        break  # Restart scan after one removal
    
    # Final greedy removal pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(best_comps) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = best_comps[:idx] + best_comps[idx+1:]
            is_valid, _ = _check_network_fast(candidate, bits)
            if is_valid:
                best_comps = candidate
                improved = True
                break
    
    return best_comps
# EVOLVE-BLOCK-END
