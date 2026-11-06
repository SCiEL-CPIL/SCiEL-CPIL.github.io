import numpy as np
import time

N = 18

# Precompute all boolean inputs
# This is done once at module load or first call
_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = np.zeros((1 << N, N), dtype=np.uint8)
        for i in range(N):
            _ALL_INPUTS[:, i] = (np.arange(1 << N) // (1 << i)) % 2
    return _ALL_INPUTS

def _generate_bitonic_sort(n):
    comparators = []
    
    def bitonic_merge(n, offset, direction):
        if n == 1:
            return
        k = n // 2
        for i in range(k):
            a = offset + i
            b = offset + i + k
            if direction:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(k, offset, direction)
        bitonic_merge(n - k, offset + k, not direction)
        
    def sort(n, offset, direction):
        if n == 1:
            return
        k = n // 2
        sort(k, offset, direction)
        sort(n - k, offset + k, not direction)
        bitonic_merge(n, offset, direction)
        
    sort(n, 0, True)
    return comparators

def _is_valid_network(network):
    if not network:
        return N <= 1
        
    inputs = _get_all_inputs()
    data = inputs.copy()
    
    for i, j in network:
        col_i = data[:, i]
        col_j = data[:, j]
        min_val = np.minimum(col_i, col_j)
        max_val = np.maximum(col_i, col_j)
        data[:, i] = min_val
        data[:, j] = max_val
        
    return np.all(data[:, :-1] <= data[:, 1:])

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Generate initial network
    current_net = _generate_bitonic_sort(N)
    
    # Prune redundant comparators
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        # Iterate from end to start
        for idx in range(len(current_net) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing current_net[idx]
            candidate_net = current_net[:idx] + current_net[idx+1:]
            
            # Check validity
            if _is_valid_network(candidate_net):
                current_net = candidate_net
                improved = True
                break  # Restart loop with new list length
        
    return current_net
