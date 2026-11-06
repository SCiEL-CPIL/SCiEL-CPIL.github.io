N = 18

import numpy as np
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 18 wires using a layered heuristic approach.
    1. Generate a random layered network with sufficient depth.
    2. Verify correctness using a vectorized NumPy check over all 2^18 boolean inputs.
    3. Optimize by greedily removing redundant comparators and performing local search swaps.
    """
    start_time = time.time()
    timeout = 55  # seconds

    # Precompute all boolean inputs for fast checking
    # Shape: (2^N, N)
    # We use int8 to save memory and speed up operations
    num_inputs = 1 << N
    indices = np.arange(num_inputs, dtype=np.int32)
    all_inputs = np.zeros((num_inputs, N), dtype=np.int8)
    for i in range(N):
        all_inputs[:, i] = (indices >> i) & 1

    def is_sorting_network(comparators):
        """
        Check if the network sorts all boolean inputs using vectorized NumPy operations.
        """
        if not comparators:
            return True
            
        # Copy the inputs to avoid modifying the original
        # This is the main cost, but 2^18 * 18 * 1 byte ~ 4.7MB copy is fast.
        state = all_inputs.copy()
        
        for i, j in comparators:
            # Vectorized compare-exchange
            # min to i, max to j
            a = state[:, i]
            b = state[:, j]
            # np.minimum and np.maximum are vectorized
            state[:, i] = np.minimum(a, b)
            state[:, j] = np.maximum(a, b)
            
        # Check if sorted
        # A state is sorted if state[k] <= state[k+1] for all k
        # We can check if any pair is inverted
        for k in range(N - 1):
            if np.any(state[:, k] > state[:, k + 1]):
                return False
        return True

    def generate_random_layered_network(depth):
        """
        Generate a random sorting network candidate with fixed depth.
        Each layer is a random matching (permutation of wires, paired 0-1, 2-3, etc.)
        """
        comparators = []
        for _ in range(depth):
            # Random permutation of wires
            perm = list(range(N))
            random.shuffle(perm)
            # Pair up adjacent elements in the permutation
            for k in range(0, N, 2):
                i = perm[k]
                j = perm[k + 1]
                if i > j:
                    i, j = j, i
                comparators.append((i, j))
        return comparators

    # 1. Find an initial valid network
    current = None
    best = None
    best_len = float('inf')
    
    # Try increasing depths until we find a valid network
    # A random network of depth N is often valid, but not guaranteed.
    # Depth 20 is likely safe.
    for depth in range(N, N + 10):
        for attempt in range(10):
            if time.time() - start_time > timeout:
                break
            candidate = generate_random_layered_network(depth)
            if is_sorting_network(candidate):
                current = candidate
                best = list(candidate)
                best_len = len(candidate)
                break
        if current:
            break
            
    if current is None:
        # Fallback: Use Odd-Even Transposition if random fails (should not happen)
        # But we want to avoid reusing design? OET is a standard baseline.
        # Let's just try a very deep random network.
        candidate = generate_random_layered_network(30)
        if is_sorting_network(candidate):
            current = candidate
            best = list(candidate)
            best_len = len(candidate)
        else:
            # Emergency fallback: dense network
            # This is unlikely to be needed
            current = []
            for i in range(N):
                for j in range(i+1, N):
                    current.append((i, j))
            best = list(current)
            best_len = len(current)

    # 2. Optimization: Greedy Removal
    # We try to remove each comparator. If the network remains valid, we keep it removed.
    # We iterate until no more removals are possible.
    
    removed_any = True
    while removed_any and (time.time() - start_time < timeout):
        removed_any = False
        # Shuffle order to break symmetries
        indices = list(range(len(current)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            
            # Try removing comparator at idx
            comp_to_remove = current[idx]
            new_comps = current[:idx] + current[idx+1:]
            
            if is_sorting_network(new_comps):
                current = new_comps
                removed_any = True
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
                    # Optional: break early if we hit a very low count?
                    # No, keep trying to remove more.

    # 3. Local Search: Swap/Mutate
    # Try to swap a comparator with another one at a different position.
    # Or replace a comparator with a different one.
    
    for _ in range(100): # Limit iterations
        if time.time() - start_time > timeout:
            break
        
        if not current:
            break
            
        # Pick a random comparator to modify
        idx = random.randint(0, len(current) - 1)
        old_comp = current[idx]
        
        # Generate a new random comparator
        i, j = random.sample(range(N), 2)
        if i > j:
            i, j = j, i
        
        new_comp = (i, j)
        
        # If it's the same, skip
        if new_comp == old_comp:
            continue
            
        # Try replacing
        new_comps = list(current)
        new_comps[idx] = new_comp
        
        if is_sorting_network(new_comps):
            current = new_comps
            if len(current) < best_len:
                best = list(current)
                best_len = len(current)
                
            # After a successful swap, try to remove redundancies again
            # This might be expensive, so do it sparingly
            if random.random() < 0.5:
                # Simple greedy removal pass
                for r_idx in range(len(current)):
                    if time.time() - start_time > timeout:
                        break
                    test_comps = current[:r_idx] + current[r_idx+1:]
                    if is_sorting_network(test_comps):
                        current = test_comps
                        if len(current) < best_len:
                            best = list(current)
                            best_len = len(current)

    return best
# EVOLVE-BLOCK-END
