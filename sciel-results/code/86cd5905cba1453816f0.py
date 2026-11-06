import time
import numpy as np

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit, 10s margin
    
    # 1. Generate initial Bitonic Sort Network
    comparators = []
    
    def bitonic_sort(start, size, direction):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(start, half, direction)
        bitonic_sort(start + half, half, not direction)
        bitonic_merge(start, size, direction)

    def bitonic_merge(start, size, direction):
        if size <= 1:
            return
        step = size // 2
        while step >= 1:
            for i in range(start, start + size, 2 * step):
                for j in range(step):
                    a = i + j
                    b = i + j + step
                    comparators.append((a, b))
            step //= 2

    bitonic_sort(0, N, True)
    
    # 2. Prepare for Greedy Removal
    # Generate all 2^N inputs
    # Using vectorized bit extraction
    total_inputs = 1 << N
    inputs = np.zeros((total_inputs, N), dtype=np.uint8)
    
    # Pre-compute arange to avoid recreation
    indices = np.arange(total_inputs)
    for j in range(N):
        inputs[:, j] = (indices >> j) & 1
        
    def simulate(comps):
        state = inputs.copy()
        for i, j in comps:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        return state

    def is_sorted(state):
        # Check if all rows are non-decreasing
        # For binary inputs, this means no 1->0 transition
        diffs = np.diff(state, axis=1)
        return np.all(diffs >= 0)

    current_comps = comparators.copy()
    
    # Greedy Removal Loop
    # We try to remove comparators.
    # We do multiple passes until no more can be removed or time is up.
    
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        # Iterate backwards
        for idx in range(len(current_comps) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Temporarily remove
            test_comps = current_comps[:idx] + current_comps[idx+1:]
            
            # Simulate
            state = simulate(test_comps)
            
            if is_sorted(state):
                # Keep it removed
                current_comps.pop(idx)
                removed_any = True
                # Note: We don't break the for loop, we continue to check previous indices
                # But the list has changed, so `idx` now points to the next element to the left.
    
    return current_comps
