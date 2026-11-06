import numpy as np
import time

N = 16

# EVOLVE-BLOCK-START
def _bitonic_merge_network(n):
    """Generate comparators for bitonic merge of size n."""
    comps = []
    k = n // 2
    while k >= 1:
        for i in range(n - k):
            comps.append((i, i + k))
        k //= 2
    return comps

def _bitonic_sort_network(n):
    """Generate Batcher's Bitonic Sort network for n wires."""
    if n == 1:
        return []
    half = n // 2
    # Left half: ascending
    left = _bitonic_sort_network(half)
    
    # Right half: descending
    right_asc = _bitonic_sort_network(half)
    right_desc = []
    for i, j in right_asc:
        i_new = half - 1 - j
        j_new = half - 1 - i
        right_desc.append((half + i_new, half + j_new))
        
    # Merge step
    merge = _bitonic_merge_network(n)
    
    return left + right_desc + merge

def construct():
    net = _bitonic_sort_network(N)
    
    # Precompute inputs for validity check
    inputs = np.array(np.unravel_index(np.arange(2**N), (2,)*N)).T
    
    def is_valid(network):
        wires = inputs.copy()
        for i, j in network:
            min_val = np.minimum(wires[i], wires[j])
            max_val = np.maximum(wires[i], wires[j])
            wires[i] = min_val
            wires[j] = max_val
        diffs = np.diff(wires, axis=0)
        return np.all(diffs >= 0)
    
    best_net = net
    best_count = len(net)
    
    start_time = time.time()
    timeout = 85.0 # Safety margin
    
    # Hill Climbing with Swaps
    # We try to remove comparators. If stuck, we swap to find new removal opportunities.
    
    while time.time() - start_time < timeout:
        improved = False
        
        # 1. Try Removals
        indices = list(range(len(best_net)))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            if idx >= len(best_net):
                continue
                
            candidate = best_net[:idx] + best_net[idx+1:]
            if is_valid(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break # Restart loop with new net
                
        if improved:
            continue
            
        # 2. If no removal, try Swaps
        # Try a limited number of random swaps to escape local minima
        # If a swap keeps it valid, accept it.
        # This doesn't reduce count, but might enable future reductions.
        
        swap_found = False
        num_swaps_to_try = 50 # Adjust based on time
        
        for _ in range(num_swaps_to_try):
            if time.time() - start_time > timeout:
                break
            if len(best_net) < 2:
                break
                
            i, j = np.random.choice(len(best_net), 2, replace=False)
            candidate = best_net[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
            if is_valid(candidate):
                # Accept swap
                best_net = candidate
                swap_found = True
                break # Restart loop to try removals again
                
        if not swap_found:
            # If no swaps work either, we are likely at a global minimum for this topology
            # or we are out of time/luck.
            # Try a few more random restarts? No, stick to one topology.
            break
            
    return best_net
# EVOLVE-BLOCK-END
