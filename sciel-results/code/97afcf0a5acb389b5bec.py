import time
import numpy as np

N = 17
TOTAL_MASKS = 1 << N  # 131072

# EVOLVE-BLOCK-START
def _run_network(net, masks):
    """Run network `net` (list of (i,j)) on array of masks (shape (M,)).
    Returns array of sorted masks (shape (M,))."""
    if len(net) == 0:
        return masks.astype(np.int32).copy()
    
    x = masks.astype(np.int32).copy()
    # Optimization: Process in batches if possible, but for now standard loop
    # Vectorized comparator application
    for i, j in net:
        a = (x >> i) & 1
        b = (x >> j) & 1
        swap = (a & ~b)  # a=1, b=0
        x = (x & ~(swap << i)) | (swap << j)
    return x

def _count_unsorted(net, masks):
    """Count how many masks are not sorted by net."""
    if len(net) == 0:
        # Count masks that are not already sorted (i.e., not of form 00...011...1)
        # A mask is sorted if it is of the form (2^k - 1) << (N-k)
        # Actually, easier: run a trivial check or just return len(masks) if we assume unsorted
        # But for greedy, we need to know which are unsorted.
        # Let's just run the network logic. If net is empty, sorted_masks = masks.
        # So unsorted = masks that are not in sorted order.
        # Sorted order for a bitmask means all 1s are on the left (high indices) or right?
        # Task says: min to wire i, max to j. 
        # If i < j, min goes to i (lower index), max goes to j (higher index).
        # So sorted order is 0...0 1...1.
        # Let's verify: input 10 (2), N=2. i=0, j=1. a=0, b=1. swap=0. x=10 (2).
        # Input 01 (1). a=1, b=0. swap=1. x = 01 & ~1 | 1<<1 = 00 | 10 = 10 (2).
        # So sorted output is 10...10? No, 1s on the right (high index) is max value.
        # Wait. min to i, max to j.
        # If a=1, b=0 (1,0), min=0, max=1. Result: 0 at i, 1 at j.
        # So 1s move to higher indices.
        # Sorted mask looks like 00...011...1.
        
        # For empty net, a mask is "unsorted" if it doesn't match this pattern.
        # Pattern: lower k bits are 1, upper N-k bits are 0.
        # Values: 2^k - 1 for k=0..N.
        sorted_vals = set([ (1 << k) - 1 for k in range(N+1) ])
        return int(np.sum(~np.isin(masks, list(sorted_vals))))

    sorted_masks = _run_network(net, masks)
    return int(np.sum(sorted_masks != masks))

def _get_unsorted_mask_indices(net, masks):
    """Return boolean array of which masks are unsorted."""
    if len(net) == 0:
        sorted_vals = set([ (1 << k) - 1 for k in range(N+1) ])
        return ~np.isin(masks, list(sorted_vals))
    sorted_masks = _run_network(net, masks)
    return sorted_masks != masks

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    masks = np.arange(TOTAL_MASKS, dtype=np.int32)
    
    # Start with seed
    best_net = _build_seed()
    best_count_unsorted = _count_unsorted(best_net, masks)
    
    if best_count_unsorted == 0:
        best_valid = best_net[:]
    else:
        best_valid = None
        
    best_score = len(best_net) if best_valid else 999999
    
    # Precompute all possible comparator positions
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Strategy: Layered Greedy Construction
    # We will try to build a network that is efficient.
    # Instead of removing from seed, we can try to optimize the seed.
    
    # First, ensure we have a valid baseline.
    if best_valid is None:
        # Fallback to seed if it was invalid (shouldn't happen)
        return _build_seed()
        
    # Local Search Phase 1: Aggressive Removal from Seed
    # The seed is 136 comparators. We want to get down to ~71.
    # Simple removal is too slow if done naively. 
    # However, we can use a "batch removal" or "randomized removal" approach.
    
    # Let's try a different local search: 
    # 1. Randomly shuffle the network (preserving valid prefixes? No, just shuffle list)
    # 2. Try to remove any comparator that keeps the network valid.
    # 3. Repeat.
    
    # But shuffling breaks the "prefix" property if we just shuffle the list.
    # A sorting network is valid if every prefix is a valid sub-network? No.
    # The whole network must sort.
    
    # Let's stick to the removal strategy but make it faster.
    # The bottleneck is _count_unsorted.
    
    # Optimization: 
    # If we remove a comparator, we only need to check if the masks that were 
    # "fixed" by that comparator are still fixed by the rest.
    # This is complex.
    
    # Alternative: Simulated Annealing / Random Restart
    # Start with seed.
    # Move: Remove a random comparator.
    # If valid, keep.
    # If invalid, try to fix by adding a random comparator?
    
    # Let's try a "Greedy Layered" approach on the seed.
    # Group seed into layers (depths).
    # For each layer, try to remove comparators that are redundant given the previous layers.
    
    # Actually, the most effective known heuristic for small N is:
    # 1. Start with a valid network.
    # 2. Iterate:
    #    a. Find a comparator that can be removed while maintaining validity.
    #    b. Remove it.
    #    c. If none found, try to replace a comparator with a "better" one?
    
    # Given the 60s limit, we can only afford ~50-100 full verifications of 131k masks if we are fast.
    # _count_unsorted takes ~50ms for 136 comparators?
    # 131072 * 136 ops. Vectorized numpy is fast.
    # Let's benchmark mentally: 131072 * 100 ops = 13M ops. Numpy does ~100M ops/s. So ~100ms per check.
    # 55s / 100ms = 550 checks.
    
    # So we can do ~500 removal checks.
    # If we remove 1 comparator per check, we can remove 500 comparators?
    # Seed is 136. So we can check every single comparator in the seed multiple times.
    
    # Algorithm:
    # 1. net = seed
    # 2. Loop:
    #    - For each idx in reversed(net):
    #        - Check if net without idx is valid.
    #        - If yes, remove it, break inner loop, restart outer.
    #    - If no removal possible, break.
    
    # This is what the current code does, but it might be stuck or slow.
    # The current code breaks after the first successful removal.
    # Let's optimize the verification.
    
    # We can pre-compute the "state" after each prefix?
    # No, removing a middle element changes the state for all subsequent elements.
    
    # Let's try a different removal order: Random.
    # Or, try to remove comparators that are "far apart" or "redundant".
    
    # Let's implement the standard removal loop but with a time guard.
    
    net = best_valid[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing from end to start (often easier to remove later ones if they are redundant)
        # Or random order?
        # Let's try all indices.
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                break
            trial = net[:idx] + net[idx+1:]
            cnt = _count_unsorted(trial, masks)
            if cnt == 0:
                net = trial
                improved = True
                if len(net) < best_score:
                    best_score = len(net)
                    best_valid = net[:]
                break  # Restart scan from the end
        
    # If we have time, try to "swap" comparators to enable further removals?
    # Or try to add a comparator that allows removing two others? (3-for-1 exchange)
    
    # Let's try a "2-for-1" or "3-for-1" heuristic if we have time.
    # If we can't remove any single comparator, maybe we can replace one with two that enable a removal?
    # This is expensive.
    
    # Alternative: Try to construct a smaller network from scratch using a known good pattern?
    # For N=17, the best known is 71.
    # If we get stuck at 80, we are done.
    
    # Let's add a "random restart" local search.
    # 1. Take current best.
    # 2. Randomly remove 5-10 comparators (making it invalid).
    # 3. Try to fix it by adding comparators from a candidate set?
    # This is like a repair heuristic.
    
    # Given the time, simple removal is the most reliable.
    # To beat 1.0 (136), we just need to remove anything.
    # The current code likely already does this.
    # Why is it stuck at 1.0?
    # Maybe it's not removing anything?
    # Or maybe the seed is already minimal for this specific structure?
    # No, odd-even transposition is not optimal.
    
    # Let's check if the current code's removal logic is flawed.
    # It checks `if cnt == 0`.
    # If the seed is valid, and we remove one, it becomes invalid.
    # So it shouldn't remove it.
    # It should find a redundant one.
    
    # Perhaps the issue is that the seed has NO redundant comparators?
    # That is highly unlikely for 136 vs 71.
    
    # Wait, the current code returns `best_valid`.
    # If `best_valid` is the seed (136), score is 136/136 = 1.0.
    # If it removes 10, score is 126/136 < 1.0.
    
    # So the current code IS removing comparators.
    # Why is the score reported as 1.0000?
    # Maybe the evaluation is just showing the seed score because the improvement is marginal or the code is slow and times out before removing many?
    # Or maybe the "score" in the prompt is "seed/value", so 1.0 means value=seed.
    # If I improve to 100, score is 136/100 = 1.36.
    
    # So my goal is to get as low a comparator count as possible.
    
    # Let's try to be smarter about removal.
    # Instead of checking every index, we can prioritize indices.
    # Comparators in the middle of the network are more likely to be redundant?
    # Or comparators that are "close" to others?
    
    # Let's try a "randomized removal" with replacement.
    
    # Actually, let's try a different starting point.
    # What if we start with a smaller, valid network?
    # We don't have one.
    
    # Let's stick to optimizing the seed.
    
    # One optimization: 
    # When checking if `trial` is valid, we can early-exit if we find one unsorted mask.
    # `_count_unsorted` returns the count.
    # We can modify it to return False as soon as it finds a mismatch.
    
    def _is_valid(net, masks):
        if len(net) == 0:
            sorted_vals = set([ (1 << k) - 1 for k in range(N+1) ])
            return np.all(np.isin(masks, list(sorted_vals)))
        
        x = masks.astype(np.int32).copy()
        for i, j in net:
            a = (x >> i) & 1
            b = (x >> j) & 1
            swap = (a & ~b)
            x = (x & ~(swap << i)) | (swap << j)
            # Early exit check? 
            # We can't check validity until the end.
            # But we can check if x is already sorted? No, it might unsort later.
            # So no early exit in the middle.
            
        return np.all(x == masks)
    
    # This is the same cost.
    
    # Let's try to use the fact that we only need to check a SUBSET of masks?
    # No, correctness requires all.
    
    # Let's try to parallelize the checks? 
    # We can check multiple candidate removals in parallel?
    # Numpy doesn't easily allow checking 100 different networks in one go without memory blowup.
    
    # Let's just run the removal loop as fast as possible.
    
    # Re-implementing the main loop with tight time checks.
    
    net = best_valid[:]
    # Use a list of indices to try, shuffled?
    # No, deterministic is better for reproducibility.
    
    while time.time() < deadline:
        removed = False
        # Try removing from the end
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                break
            trial = net[:idx] + net[idx+1:]
            if _is_valid(trial, masks):
                net = trial
                removed = True
                if len(net) < best_score:
                    best_score = len(net)
                    best_valid = net[:]
                break
        
        if not removed:
            # Try removing from the beginning?
            # Or try a random index?
            # If we can't remove from the end, maybe we can from the start?
            # Unlikely to make a difference.
            
            # Try "swap" removal?
            # If we can't remove any single comparator, we are stuck.
            # Break to save time.
            break
            
    return best_valid

# EVOLVE-BLOCK-END
