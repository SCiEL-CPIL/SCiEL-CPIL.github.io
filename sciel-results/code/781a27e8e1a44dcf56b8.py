import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute inputs
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = ((_indices[:, None] & _bits[None, :]) > 0).astype(np.uint8)
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Sample for fast check
_SAMPLE_SIZE = 8192
_rng = np.random.RandomState(42)
_SAMPLE_INDICES = _rng.choice(_NUM_INPUTS, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]

def simulate_layered(comparators, inputs):
    """
    Simulate comparators, batching disjoint comparators into layers.
    """
    state = inputs.copy()
    n = len(comparators)
    i = 0
    while i < n:
        # Find the maximal set of disjoint comparators starting at i
        used = [False] * N
        layer = []
        j = i
        while j < n:
            a, b = comparators[j]
            if not used[a] and not used[b]:
                used[a] = True
                used[b] = True
                layer.append((a, b))
                j += 1
            else:
                break
        
        # Apply layer
        if len(layer) == 1:
            a, b = layer[0]
            min_val = np.minimum(state[:, a], state[:, b])
            max_val = np.maximum(state[:, a], state[:, b])
            state[:, a] = min_val
            state[:, b] = max_val
        elif len(layer) > 1:
            # Batch update
            # We need to be careful not to overwrite values needed for other comparators in the same layer.
            # Since they are disjoint, we can compute all mins and maxs first, then assign.
            mins = np.empty((len(inputs), len(layer)), dtype=np.uint8)
            maxs = np.empty((len(inputs), len(layer)), dtype=np.uint8)
            for k, (a, b) in enumerate(layer):
                mins[:, k] = np.minimum(state[:, a], state[:, b])
                maxs[:, k] = np.maximum(state[:, a], state[:, b])
            for k, (a, b) in enumerate(layer):
                state[:, a] = mins[:, k]
                state[:, b] = maxs[:, k]
        
        i = j
    return state

def is_sorted_network_full(comparators):
    if len(comparators) > 4096:
        return False
    state = simulate_layered(comparators, _INPUTS)
    return bool(np.array_equal(state, _SORTED_INPUTS))

def is_sorted_network_sample(comparators):
    if len(comparators) > 4096:
        return False
    state = simulate_layered(comparators, _SAMPLE_INPUTS)
    return bool(np.array_equal(state, _SAMPLE_SORTED))

def oe_merge_sort(n):
    """Odd-even merge sort network for n elements."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    left = oe_merge_sort(half)
    right = oe_merge_sort(n - half)
    right = [(i + half, j + half) for i, j in right]
    
    # Odd-even merge
    merge = []
    for phase in range(n - 1):
        offset = phase % 2
        for i in range(offset, n - 1, 2):
            merge.append((i, i + 1))
            
    return left + right + merge

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Initialize with a valid network
    # Odd-even merge sort is a good starting point, but bitonic might be denser and prune better?
    # Let's try Odd-Even Merge Sort first.
    current = oe_merge_sort(N)
    
    if not is_sorted_network_full(current):
        # Fallback: Odd-even transposition
        current = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                current.append((i, i + 1))
        
    best_network = list(current)
    best_len = len(current)
    
    # 2. Pruning Loop
    # We try to remove each comparator.
    # To speed up, we first check on a sample.
    
    # We will do multiple passes.
    while time.time() < deadline:
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 5.0:
                break
            
            candidate = current.copy()
            candidate.pop(i)
            
            # Fast check
            if is_sorted_network_sample(candidate):
                # Full check
                if is_sorted_network_full(candidate):
                    current = candidate
                    best_network = candidate
                    best_len = len(current)
                    progress = True
                    # Do not increment i, check the new position i again
                    continue
            
            i += 1
        
        if not progress:
            # No more deletions possible in this pass
            break
    
    # 3. Local Search: Swapping
    # Sometimes swapping comparators allows for further deletion.
    # We try swapping adjacent comparators and then attempt to delete one of them or others.
    # This is expensive, so we only do it if we have time.
    
    if time.time() < deadline - 10.0:
        # Try a few random swaps and then re-prune
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            if len(current) < 2:
                break
            
            idx = random.randint(0, len(current) - 1)
            if idx > 0:
                swap_idx = idx - 1
            elif idx < len(current) - 1:
                swap_idx = idx + 1
            else:
                continue
                
            candidate = current.copy()
            candidate[idx], candidate[swap_idx] = candidate[swap_idx], candidate[idx]
            
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    # Accept swap if it leads to a smaller network after re-pruning
                    # For now, just accept if valid, then re-run pruning
                    current = candidate
                    # Re-prune briefly
                    i = 0
                    while i < len(current) and time.time() < deadline - 5.0:
                        cand2 = current.copy()
                        cand2.pop(i)
                        if is_sorted_network_sample(cand2):
                            if is_sorted_network_full(cand2):
                                current = cand2
                                best_network = current
                                best_len = len(current)
                                continue
                        i += 1
                    if len(current) < best_len:
                        best_network = list(current)
                        best_len = len(current)
            # If swap didn't lead to immediate improvement, revert or keep?
            # Keeping it might help future deletions, but if it didn't reduce size,
            # we might just be wandering. Let's only keep if it reduced size or
            # we are stuck. Actually, let's just keep the best found so far.
            # The `best_network` tracks the absolute best.
            # We should probably revert `current` if the swap didn't lead to a better `best_len`
            # to avoid drifting into a larger or equal network that is harder to prune.
            # However, the swap itself is valid.
            # Let's stick to the best found.
    
    # Ensure we return a valid network
    if not is_sorted_network_full(best_network):
        # This should not happen if we only update best_network when valid
        return oe_merge_sort(N)
        
    return best_network
# EVOLVE-BLOCK-END
