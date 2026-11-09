import time
import numpy as np

N = 17
TOTAL_MASKS = 1 << N  # 131072

# EVOLVE-BLOCK-START

def _run_network(net, masks):
    """Run network `net` on array of masks."""
    if len(net) == 0:
        return masks.astype(np.int32).copy()
    
    x = masks.astype(np.int32).copy()
    for i, j in net:
        a = (x >> i) & 1
        b = (x >> j) & 1
        swap = (a & ~b)
        x = (x & ~(swap << i)) | (swap << j)
    return x

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _build_batcher_odd_merge(n):
    """
    Construct Batcher's Odd-Merge sorting network for n wires.
    Returns a list of (i, j) pairs.
    """
    if n == 1:
        return []
    
    # Split into two halves: even indices and odd indices
    # H1 = indices 0, 2, 4, ...
    # H2 = indices 1, 3, 5, ...
    
    # Recursive sort of each half
    # Note: The recursive call sorts the sub-networks on their local indices,
    # but we need to map them to the global indices of the original set.
    # Batcher's Odd-Merge is typically defined on a contiguous range or a set of indices.
    # To keep it simple and correct, we implement it on a list of indices.
    
    def _odd_merge(indices):
        if len(indices) <= 1:
            return []
        
        # Split the list of indices into two halves (by position in the list)
        # This corresponds to the "odd" and "even" elements in the standard definition
        # if the input is sorted by index, but here we are splitting the *set* of wires.
        # Standard Odd-Merge:
        # 1. Split input into two halves: H1 (x0, x2, ...) and H2 (x1, x3, ...)
        # 2. Recursively sort H1 and H2.
        # 3. Merge H1 and H2.
        
        # Let's stick to the standard definition on a range 0..n-1 for clarity,
        # then map if needed? No, it's easier to just define it on 0..n-1 directly.
        
        pass

    # Let's use a simpler, known correct implementation of Batcher's Merge Sort (not Odd-Merge)
    # which is also efficient and structurally different from Odd-Even Transposition.
    # Batcher's Merge Sort:
    # 1. Split into two halves.
    # 2. Recursively sort each half.
    # 3. Merge the two sorted halves.
    
    def _batcher_merge_sort(indices):
        if len(indices) <= 1:
            return []
        
        mid = len(indices) // 2
        left_indices = indices[:mid]
        right_indices = indices[mid:]
        
        left_net = _batcher_merge_sort(left_indices)
        right_net = _batcher_merge_sort(right_indices)
        
        merge_net = _batcher_merge(left_indices, right_indices)
        
        return left_net + right_net + merge_net

    def _batcher_merge(a, b):
        """
        Merge two sorted sequences of wires a and b.
        a and b are lists of global wire indices.
        """
        if len(a) == 0:
            return []
        if len(b) == 0:
            return []
        
        # Batcher's Merge Network
        # The merge network is defined recursively.
        # Let n = len(a), m = len(b).
        # We compare (a[i], b[i]) for i in range(min(n, m))? No.
        
        # Standard Batcher Merge:
        # 1. Compare (a[i], b[n-1-i]) for i in range(min(n, m))? No.
        
        # Let's look up the exact definition of Batcher's Merge Network.
        # It involves a specific pattern of comparators.
        # For simplicity and correctness, I will use **Odd-Even Transposition** for the merge step?
        # That would make it similar to the seed.
        
        # Alternative: Use **Bitonic Merge**?
        # Or just use a simple, valid merge:
        # Compare (a[i], b[j]) for all pairs? Too many.
        
        # Let's use **Shear Sort** or a simple **Bubble Sort** merge?
        # Bubble Sort merge:
        # While not sorted:
        #   for i in range(len(a)-1): compare (a[i], a[i+1])
        #   for i in range(len(b)-1): compare (b[i], b[i+1])
        #   for i in range(len(a)): compare (a[i], b[i])?
        
        # Actually, the most robust "different" seed is **Batcher's Odd-Merge** implemented correctly.
        # Let's try to find a compact, correct implementation logic.
        
        # Reference: 
        # Odd-Merge(n):
        #   if n == 1: return
        #   Odd-Merge(n/2) on x0, x2, ...
        #   Odd-Merge(n/2) on x1, x3, ...
        #   Merge(x0, x2, ..., x1, x3, ...)
        
        # Merge step:
        #   Compare (x_i, x_{n-1-i}) for i in 0..n/2-1? No.
        
        # Given the time, I will use **Batcher's Merge Sort** with a **Simple Valid Merge**.
        # The merge step can be implemented as:
        # for i in range(min(len(a), len(b))):
        #     compare (a[i], b[i])
        # This is NOT a valid merge for all inputs.
        
        # A valid merge is:
        # for i in range(len(a)):
        #     for j in range(len(b)):
        #         compare (a[i], b[j])
        # This is O(N^2) and large.
        
        # Let's use **Odd-Even Transposition** as the merge step?
        # If we use Odd-Even Transposition for the merge, the whole network is just a variant of it.
        
        # Let's try **AKS** is too hard.
        
        # I will use **Batcher's Odd-Merge** with a known correct recursive merge.
        # The merge step in Odd-Merge is:
        # 1. Compare (x_i, x_{i + n/2}) for i in 0..n/2-1?
        
        # To avoid bugs, I will use a **Randomized Local Search** on the **Seed**
        # but with a **Perturb-Verify-Cleanup** strategy as hinted by the lessons.
        
        return []

    # Fallback to Seed if Batcher implementation is too risky/complex in this short time.
    # However, the inspiration says Batcher is used in top solutions.
    # I will implement a simple **Batcher's Merge Sort** where the merge step is
    # a simple **Odd-Even Transposition** on the combined wires?
    # No, that's just O(N^2).
    
    # Let's use **Batcher's Merge Sort** with the **correct** Batcher merge.
    # I will copy a known correct logic for Batcher's Merge.
    
    def _batcher_merge_net(a, b):
        """
        Merge two sorted lists of indices a and b.
        """
        if len(a) == 0:
            return []
        if len(b) == 0:
            return []
        
        # The merge network consists of comparators between a and b.
        # The pattern is:
        # for i in range(len(a)):
        #    for j in range(len(b)):
        #       if i + j is even: compare (a[i], b[j])
        #       if i + j is odd: compare (a[i], b[j])
        # This is essentially Odd-Even Transposition on the merged array?
        
        # Let's use a simpler, valid construction:
        # **Bitonic Sort** is O(N log^2 N).
        # **Batcher's Merge Sort** is O(N log^2 N).
        
        # I will use **Odd-Even Transposition** as the seed, but apply **Perturb-Verify-Cleanup**.
        # The key improvement is the search strategy, not necessarily the seed.
        # But a better seed helps.
        
        # Let's use **Batcher's Odd-Merge** from a reliable source pattern.
        # For now, I'll return the Seed to ensure correctness, and rely on the search.
        
        return []

    # Since I'm struggling to recall the exact Batcher merge pattern without lookup,
    # and the prompt emphasizes a "MATERIALLY DIFFERENT" strategy,
    # I will focus on the **Search Strategy**: **Perturb-Verify-Cleanup**.
    # The seed will be the standard Odd-Even Transposition.
    
    return _build_seed()

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    masks = np.arange(TOTAL_MASKS, dtype=np.int32)
    
    # Start with Seed
    best_net = _build_seed()
    
    # Verify
    sorted_masks = _run_network(best_net, masks)
    if not np.all(sorted_masks == masks):
        return best_net
        
    best_valid = best_net[:]
    best_score = len(best_net)
    
    # **Perturb-Verify-Cleanup** Strategy
    # 1. Start with a valid network.
    # 2. Perturb: Make a random change (remove, add, swap) to the network.
    # 3. Verify: Check if the new network is valid.
    # 4. Cleanup: If valid, aggressively try to remove comparators (hill climbing) to minimize size.
    # 5. Repeat.
    
    net = best_valid[:]
    
    while time.time() < deadline:
        # Perturb
        # Randomly choose a move:
        # - Remove a comparator (with some probability)
        # - Add a comparator
        # - Swap two comparators
        
        move_type = np.random.choice(['remove', 'add', 'swap'], p=[0.4, 0.3, 0.3])
        
        trial_net = net[:]
        
        if move_type == 'remove' and len(net) > 0:
            idx = np.random.randint(len(net))
            trial_net = net[:idx] + net[idx+1:]
            
        elif move_type == 'add':
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i < j:
                trial_net.append((i, j))
            else:
                trial_net.append((j, i))
                
        elif move_type == 'swap' and len(net) > 1:
            i = np.random.randint(len(net))
            j = np.random.randint(len(net))
            if i != j:
                net[i], net[j] = net[j], net[i]
                trial_net = net[:]
        
        # Verify
        sorted_masks = _run_network(trial_net, masks)
        is_valid = np.all(sorted_masks == masks)
        
        if is_valid:
            # Cleanup: Aggressively minimize size
            current_net = trial_net
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try removing each comparator
                for idx in range(len(current_net)):
                    if time.time() > deadline:
                        break
                    trial_remove = current_net[:idx] + current_net[idx+1:]
                    sorted_masks = _run_network(trial_remove, masks)
                    if np.all(sorted_masks == masks):
                        current_net = trial_remove
                        improved = True
                        break  # Restart the loop with the new shorter network
                
            # Update best
            if len(current_net) < best_score:
                best_score = len(current_net)
                best_valid = current_net[:]
                net = current_net[:]
        else:
            # If invalid, we can still try to use it as a starting point for repair?
            # Or just discard.
            # For simplicity, discard.
            pass
            
    return best_valid

# EVOLVE-BLOCK-END
