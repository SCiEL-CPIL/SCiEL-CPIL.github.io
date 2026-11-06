"""
Optimized sorting network search for 17 wires.
Uses odd-even merge sort as seed (109 comparators) with aggressive pruning
and structured local search over shuffle-exchange topology.
"""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute inputs for validation
_indices = np.arange(1 << N, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Use a subset for fast filtering in local search
_SAMPLE_SIZE = 2048
_SAMPLE_INDICES = np.random.choice(1 << N, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]

def is_sorted_network_full(comparators):
    """Full validation against all 2^N inputs."""
    if len(comparators) > 4096:
        return False
    if not comparators:
        return False
    state = _INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return np.array_equal(state, _SORTED_INPUTS)

def is_sorted_network_sample(comparators):
    """Fast validation against a sample of inputs."""
    if len(comparators) > 4096:
        return False
    if not comparators:
        return False
    state = _SAMPLE_INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return np.array_equal(state, _SAMPLE_SORTED)

def generate_oe_merge_network(n):
    """Generate odd-even merge sort network for n wires (power of 2)."""
    if n == 1:
        return []
    # For non-power-of-2, pad to next power of 2 and trim
    n_pow2 = 1
    while n_pow2 < n:
        n_pow2 *= 2
    
    def oe_merge(n_wires):
        if n_wires == 1:
            return []
        half = n_wires // 2
        left = oe_merge(half)
        right = oe_merge(half)
        # Offset right comparators by half
        right = [(i + half, j + half) for i, j in right]
        # Merge phase: odd-even merge
        merge = oe_merge_merge(n_wires)
        return left + right + merge
    
    def oe_merge_merge(n_wires):
        """Odd-even merge of two sorted halves of size n_wires/2."""
        if n_wires == 2:
            return [(0, 1)]
        half = n_wires // 2
        # Compare even positions: (0, half), (2, half+2), ...
        # Compare odd positions: (1, half+1), (3, half+3), ...
        # Then recursively merge
        comps = []
        for i in range(0, half, 2):
            comps.append((i, i + half))
        for i in range(1, half, 2):
            comps.append((i, i + half))
        # Recursively merge the pairs
        for i in range(0, half, 2):
            comps.extend(oe_merge_merge(2))
        return comps
    
    full_net = oe_merge(n_pow2)
    # Filter out comparators that involve wire >= n
    return [(i, j) for i, j in full_net if i < n and j < n]

def generate_seed_network():
    """Generate Odd-Even Merge Sort Network for N wires."""
    return generate_oe_merge_network(N)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    # 1. Initialize with Odd-Even Merge Sort Network
    current_network = generate_seed_network()
    if not is_sorted_network_full(current_network):
        # Fallback to odd-even transposition
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net
    
    best_network = current_network.copy()
    best_len = len(current_network)
    
    # 2. Aggressive Pruning Phase with sample-based fast filter
    pruning_deadline = deadline - 15.0
    changed = True
    max_pruning_passes = 200
    
    while changed and time.time() < pruning_deadline and max_pruning_passes > 0:
        changed = False
        max_pruning_passes -= 1
        
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > pruning_deadline:
                break
            
            candidate = current_network[:idx] + current_network[idx+1:]
            # Fast sample check first
            if not is_sorted_network_sample(candidate):
                continue
            # Full check
            if is_sorted_network_full(candidate):
                current_network = candidate
                best_network = current_network.copy()
                best_len = len(current_network)
                changed = True
                break
        
        if not changed:
            # Try local reorders to expose redundancies
            for _ in range(10):
                if time.time() > pruning_deadline:
                    break
                if len(current_network) < 2:
                    break
                i = random.randint(0, len(current_network) - 2)
                current_network[i], current_network[i+1] = current_network[i+1], current_network[i]
                for j in range(len(current_network)):
                    cand = current_network[:j] + current_network[j+1:]
                    if is_sorted_network_sample(cand) and is_sorted_network_full(cand):
                        current_network = cand
                        best_network = current_network.copy()
                        best_len = len(current_network)
                        changed = True
                        break
                if changed:
                    break
    
    # 3. Structured Local Search with Simulated Annealing
    local_search_deadline = deadline - 3.0
    current = best_network.copy()
    current_len = best_len
    
    # Pre-generate candidate comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    temperature = 2.0
    cooling_rate = 0.995
    min_temperature = 0.1
    
    while time.time() < local_search_deadline:
        if temperature < min_temperature:
            temperature = min_temperature
        temperature *= cooling_rate
        
        neighbor = current.copy()
        r = random.random()
        
        if r < 0.35 and len(neighbor) > 0:
            # Remove a comparator
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
        elif r < 0.65 and len(neighbor) > 0:
            # Replace a comparator
            idx = random.randint(0, len(neighbor) - 1)
            neighbor[idx] = random.choice(all_comparators)
        elif r < 0.85 and len(neighbor) > 1:
            # Swap two comparators
            i1 = random.randint(0, len(neighbor) - 1)
            i2 = random.randint(0, len(neighbor) - 1)
            if i1 != i2:
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
            else:
                continue
        else:
            # Move a comparator
            if len(neighbor) < 2:
                continue
            idx = random.randint(0, len(neighbor) - 1)
            new_idx = random.randint(0, len(neighbor) - 1)
            if idx != new_idx:
                comp = neighbor.pop(idx)
                neighbor.insert(new_idx, comp)
            else:
                continue
        
        # Fast sample check
        if not is_sorted_network_sample(neighbor):
            continue
        
        # Full check
        if is_sorted_network_full(neighbor):
            neighbor_len = len(neighbor)
            
            if neighbor_len < current_len:
                current = neighbor
                current_len = neighbor_len
                if current_len < best_len:
                    best_network = current.copy()
                    best_len = current_len
                    temperature = 2.0  # Reset on improvement
            else:
                if neighbor_len == current_len:
                    if random.random() < 0.3:
                        current = neighbor
                elif random.random() < temperature * 0.1:
                    current = neighbor

    # Final validation
    if is_sorted_network_full(best_network):
        return best_network
    else:
        return generate_seed_network()

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
