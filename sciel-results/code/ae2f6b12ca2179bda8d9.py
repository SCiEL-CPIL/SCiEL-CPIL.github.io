import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Batcher's bitonic sort network for 16 wires: 60 comparators
    def bitonic_network(n):
        """Generate Batcher's bitonic sorting network for n wires."""
        comparators = []
        def sort_stage(k, start, size, ascending):
            if size == 1:
                return
            half = size // 2
            # First sort halves in opposite directions to create bitonic sequence
            sort_stage(k + 1, start, half, not ascending)
            sort_stage(k + 1, start + half, half, ascending)
            # Then merge the bitonic sequence
            merge_stage(k + 1, start, half, ascending)
        
        def merge_stage(k, start, size, ascending):
            if size == 1:
                return
            half = size // 2
            # Compare elements distance 'size' apart
            for i in range(half):
                a = start + i
                b = start + i + half
                if ascending:
                    comparators.append((a, b))
                else:
                    comparators.append((b, a))
            # Recursively merge halves
            merge_stage(k + 1, start, half, ascending)
            merge_stage(k + 1, start + half, half, ascending)
        
        sort_stage(0, 0, n, True)
        return comparators
    
    initial_net = bitonic_network(N)
    
    # Pre-compute inputs and targets
    inputs = np.arange(2**N, dtype=np.uint32)
    popcounts = np.array([bin(x).count('1') for x in range(2**N)], dtype=np.uint8)
    targets = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in popcounts], dtype=np.uint32)

    def verify(net):
        masks = inputs.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return np.array_equal(masks, targets)

    def verify_removals(net, indices_to_try):
        """Try removing each index and check validity.
        Returns set of indices that can be removed."""
        valid_removals = set()
        for idx in indices_to_try:
            candidate = net[:idx] + net[idx+1:]
            masks = inputs.copy()
            for i, j in candidate:
                bit_i = 1 << i
                bit_j = 1 << j
                cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
                if np.any(cond):
                    masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
            if np.array_equal(masks, targets):
                valid_removals.add(idx)
        return valid_removals

    # Verify initial
    if not verify(initial_net):
        # Fallback to odd-even transposition
        initial_net = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
        if not verify(initial_net):
            return initial_net

    current_net = initial_net
    best_net = initial_net
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 55.0
    
    random.seed(42)
    
    # Strategy: iteratively try to remove comparators
    # Bias: prefer removing comparators with larger wire distance (more likely redundant)
    # and those in the middle of the network
    while time.time() - start_time < deadline:
        net_len = len(current_net)
        if net_len <= 1:
            break
        
        # Score each comparator by "removability likelihood":
        # Higher distance between wires = more likely redundant
        # Middle positions = more likely redundant
        scored_indices = []
        for idx, (i, j) in enumerate(current_net):
            wire_dist = j - i
            pos_score = 1.0 - abs(idx - net_len / 2.0) / (net_len / 2.0)
            score = wire_dist * 0.7 + pos_score * 0.3 + random.random() * 0.5
            scored_indices.append((score, idx))
        
        # Sort by score descending (try most likely removable first)
        scored_indices.sort(key=lambda x: -x[0])
        
        # Try a batch of the top-scored indices
        batch_size = min(30, net_len)
        batch = [idx for _, idx in scored_indices[:batch_size]]
        
        removable = verify_removals(current_net, batch)
        
        if removable:
            # Remove the highest-scored removable one
            removable_scores = {idx: score for score, idx in scored_indices if idx in removable}
            best_idx = max(removable_scores, key=removable_scores.get)
            current_net = current_net[:best_idx] + current_net[best_idx+1:]
            if len(current_net) < best_len:
                best_net = current_net
                best_len = len(current_net)
        else:
            # No removals found; try swap-then-remove to escape local minimum
            if net_len > 2:
                # Pick two random positions to swap
                i1 = random.randint(0, net_len - 2)
                i2 = random.randint(i1 + 1, net_len - 1)
                candidate = current_net.copy()
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                if verify(candidate):
                    # Try removing one comparator from the swapped network
                    swap_indices = list(range(len(candidate)))
                    random.shuffle(swap_indices)
                    small_batch = swap_indices[:min(10, len(candidate))]
                    rem = verify_removals(candidate, small_batch)
                    if rem:
                        idx = rem.pop()
                        candidate = candidate[:idx] + candidate[idx+1:]
                        current_net = candidate
                        if len(current_net) < best_len:
                            best_net = current_net
                            best_len = len(current_net)
                    else:
                        current_net = candidate

    return best_net
# EVOLVE-BLOCK-END
