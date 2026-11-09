import numpy as np
import random
import time

N = 17

def check_network(comparators):
    """
    Check if the given network sorts all 2^N boolean inputs.
    Returns True if valid, False otherwise.
    """
    if not comparators:
        return N <= 1
        
    # Generate all 2^N boolean inputs
    # inputs[k, i] is the i-th bit of input k
    # Using a pre-computed array for speed? 
    # 2^17 is 131072. 
    # We can generate it once and reuse? 
    # But the values change. 
    # We can use a view or copy? 
    # Copying 131072x17 array is fast.
    
    # To avoid regenerating the base array every time, we can store it globally?
    # But it's modified in place. 
    # So we must copy the initial state.
    
    # Optimization: 
    # Instead of generating the full matrix every time, 
    # we can use a bit-manipulation trick on integers?
    # No, numpy is faster for vectorized ops.
    
    # Let's pre-compute the initial states once.
    global _INITIAL_INPUTS
    if not hasattr(check_network, '_INITIAL_INPUTS'):
        check_network._INITIAL_INPUTS = (np.arange(2**N)[:, None] // 2**np.arange(N-1, -1, -1)) % 2
        
    inputs = check_network._INITIAL_INPUTS.copy()
    
    for i, j in comparators:
        if i > j:
            i, j = j, i
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    # Check if sorted
    # A row is sorted if each element is <= the next
    # diffs = np.diff(inputs, axis=1)
    # return np.all(diffs >= 0)
    
    # Faster check: 
    # The sorted boolean vector is [0, 0, ..., 0, 1, 1, ..., 1]
    # We can check if the number of 1s is preserved (it is) and if the 1s are at the end.
    # Actually, `np.all(diffs >= 0)` is O(N) per row.
    # Alternative: Check if `inputs` matches the sorted version of itself?
    # No, we know the target: 0s then 1s.
    # But the number of 1s varies per input.
    
    # The diff check is fine.
    diffs = np.diff(inputs, axis=1)
    return np.all(diffs >= 0)

def construct():
    def odd_even_network():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    def greedy_minimize(network):
        changed = True
        while changed:
            changed = False
            # Try to remove from the end first (often less critical)
            for i in range(len(network) - 1, -1, -1):
                candidate = network[:i] + network[i+1:]
                if check_network(candidate):
                    network = candidate
                    changed = True
        return network

    best_network = None
    best_size = float('inf')
    
    start_time = time.time()
    timeout = 55
    
    # 1. Initial Greedy Minimization on Odd-Even
    current = odd_even_network()
    current = greedy_minimize(current)
    if len(current) < best_size:
        best_size = len(current)
        best_network = current[:]
        
    # 2. Local Search with Restarts
    while time.time() - start_time < timeout:
        # Try to improve best_network
        candidate = best_network[:]
        
        # Random Perturbation
        move_type = random.choice(['remove', 'move', 'swap'])
        
        if move_type == 'remove' and len(candidate) > 0:
            idx = random.randint(0, len(candidate) - 1)
            candidate = candidate[:idx] + candidate[idx+1:]
            if check_network(candidate):
                candidate = greedy_minimize(candidate)
                if len(candidate) < best_size:
                    best_size = len(candidate)
                    best_network = candidate[:]
                    
        elif move_type == 'move' and len(candidate) > 1:
            idx = random.randint(0, len(candidate) - 1)
            comp = candidate.pop(idx)
            new_idx = random.randint(0, len(candidate))
            candidate.insert(new_idx, comp)
            if check_network(candidate):
                candidate = greedy_minimize(candidate)
                if len(candidate) < best_size:
                    best_size = len(candidate)
                    best_network = candidate[:]
                    
        elif move_type == 'swap' and len(candidate) > 1:
            i, j = random.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if check_network(candidate):
                candidate = greedy_minimize(candidate)
                if len(candidate) < best_size:
                    best_size = len(candidate)
                    best_network = candidate[:]

    return best_network
