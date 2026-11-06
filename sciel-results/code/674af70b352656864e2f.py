import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 16 wires using a structural approach:
    1. Start from the known optimal 60-comparator AKS-style network (Batcher's odd-even merge sort).
    2. Use a more efficient verification that can handle large batches.
    3. Perform local search: try removing comparators, swapping, and inserting to find a smaller network.
    
    The key structural change: use Batcher's odd-even mergesort as the starting point (known to be 60 comparators for N=16),
    then apply a more aggressive reduction strategy using a different representation (tracking which comparators are "essential").
    """
    
    # Batcher's odd-even mergesort network for N=16
    # This is the known optimal 60-comparator network
    def batcher_odd_even_merge_sort(n):
        """Generate Batcher's odd-even mergesort network for 2^k wires."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        # Recursively sort even and odd positions
        k = n // 2
        even_net = batcher_odd_even_merge_sort(k)
        odd_net = batcher_odd_even_merge_sort(k)
        
        # Map even indices: 0->0, 1->2, 2->4, ...
        even_mapped = [(2*i, 2*j) for i, j in even_net]
        # Map odd indices: 0->1, 1->3, 2->5, ...
        odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_net]
        
        # Merge the two halves using odd-even merge
        merge_net = batcher_odd_even_merge(k)
        
        # Combine: sort evens, sort odds, then merge
        return even_mapped + odd_mapped + merge_net
    
    def batcher_odd_even_merge(k):
        """Generate the odd-even merge comparators for merging two sorted halves of size k."""
        if k == 1:
            return [(0, 1)]
        
        # First compare (0,1), (2,3), ... (even, odd) pairs
        first = [(2*i, 2*i+1) for i in range(k)]
        
        # Recursively merge the even and odd subsequences
        k2 = k // 2
        even_merge = batcher_odd_even_merge(k2)
        odd_merge = batcher_odd_even_merge(k2)
        
        # Map even subsequence: 0->0, 1->2, 2->4, ...
        even_mapped = [(2*i, 2*j) for i, j in even_merge]
        # Map odd subsequence: 0->1, 1->3, 2->5, ...
        odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_merge]
        
        return first + even_mapped + odd_mapped
    
    # Generate the Batcher network
    batcher_net = batcher_odd_even_merge_sort(N)
    
    # Verify the Batcher network is correct
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
    
    if not verify(batcher_net):
        # Fallback to odd-even transposition if Batcher is wrong
        initial_net = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
        return initial_net
    
    current_net = batcher_net
    best_net = batcher_net
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 55.0
    
    random.seed(42)
    
    # More aggressive local search with multiple strategies
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < deadline:
        net_len = len(current_net)
        if net_len <= best_len - 10:
            break
        
        # Strategy 1: Try to remove random comparators
        # Focus on comparators that are less likely to be essential
        # Heuristic: comparators in the middle of the network, or those with similar wire indices
        
        indices = list(range(net_len))
        random.shuffle(indices)
        
        # Try a larger batch
        batch_size = min(30, net_len)
        batch = indices[:batch_size]
        
        found_removal = False
        for idx in batch:
            if time.time() - start_time > deadline:
                break
            candidate = current_net[:idx] + current_net[idx+1:]
            if verify(candidate):
                current_net = candidate
                found_removal = True
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
                    no_improve_count = 0
                break
        
        if not found_removal:
            no_improve_count += 1
            
            # Strategy 2: Try swapping two comparators
            if net_len > 1:
                i1 = random.randint(0, net_len - 1)
                i2 = random.randint(0, net_len - 1)
                if i1 != i2:
                    candidate = current_net.copy()
                    candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                    if verify(candidate):
                        current_net = candidate
                        no_improve_count = 0
            
            # Strategy 3: Try replacing one comparator with a different one
            # This can help escape local minima
            if no_improve_count > 10 and net_len > 1:
                idx = random.randint(0, net_len - 1)
                i, j = current_net[idx]
                # Try replacing with a random comparator
                new_i = random.randint(0, N - 1)
                new_j = random.randint(0, N - 1)
                if new_i < new_j:
                    candidate = current_net.copy()
                    candidate[idx] = (new_i, new_j)
                    if verify(candidate):
                        current_net = candidate
                        no_improve_count = 0
            
            # Strategy 4: If stuck, restart from best and try a different random seed
            if no_improve_count > max_no_improve:
                current_net = best_net.copy()
                no_improve_count = 0
                max_no_improve = int(max_no_improve * 1.2)
        
        # If we've been stuck for a while, try a more systematic approach
        if no_improve_count > 20:
            # Try removing all comparators that don't affect the output for a subset of inputs
            # This is a heuristic to find redundant comparators
            subset_size = min(1000, 2**N)
            subset_inputs = np.random.choice(2**N, size=subset_size, replace=False).astype(np.uint32)
            
            for idx in range(len(current_net)):
                if time.time() - start_time > deadline:
                    break
                candidate = current_net[:idx] + current_net[idx+1:]
                masks = subset_inputs.copy()
                for i, j in candidate:
                    bit_i = 1 << i
                    bit_j = 1 << j
                    cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
                    if np.any(cond):
                        masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
                # Check if the subset is still sorted
                subset_popcounts = np.array([bin(x).count('1') for x in subset_inputs], dtype=np.uint8)
                subset_targets = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in subset_popcounts], dtype=np.uint32)
                if np.array_equal(masks, subset_targets):
                    # Promising candidate, verify fully
                    if verify(candidate):
                        current_net = candidate
                        if len(current_net) < best_len:
                            best_net = current_net
                            best_len = len(current_net)
                        no_improve_count = 0
                        break
    
    return best_net
# EVOLVE-BLOCK-END
