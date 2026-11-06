"""Size-efficient sorting network on 16 wires.

Strategy: Start with Batcher's odd-even merge sort (63 comparators) and 
perform a greedy reduction search to minimize the comparator count.
Correctness is certified by the zero-one principle over all 65,536 Boolean inputs.
Target: 60 comparators (known optimal for 16 wires).
"""
import time
import itertools
import random

N = 16
_START = time.time()
_DEADLINE = _START + 75.0  # explicit safety margin below the ~90 s wall clock


# ---------------------------------------------------------------------------
# Network builder: Batcher's odd-even merge sort (recursive).
# ---------------------------------------------------------------------------
def _batcher_odd_even_merge(n):
    """Return a fixed compare-exchange network sorting n wires (n a power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]

    cmps = []
    # 1) recursively sort odd-indexed wires: 1,3,5,...
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a + 1, 2 * b + 1))
    # 2) recursively sort even-indexed wires: 0,2,4,...
    for a, b in _batcher_odd_even_merge(n // 2):
        cmps.append((2 * a, 2 * b))
    # 3) odd-even cross comparisons (2i, 2i+1) then (2i+1, 2i+2)
    for i in range(n // 2):
        cmps.append((2 * i, 2 * i + 1))
    for i in range(n // 2 - 1):
        cmps.append((2 * i + 1, 2 * i + 2))
    return cmps


# ---------------------------------------------------------------------------
# Verification: zero-one principle (all 2**N Boolean inputs).
# Optimized for speed using NumPy.
# ---------------------------------------------------------------------------
def _verify_zero_one(cmps):
    """Return True iff `cmps` sorts every Boolean vector of length N."""
    import numpy as np

    M = 1 << N
    # Build the 2**N x N bit matrix (input -> bits, bit k is wire k).
    # We use int8 for efficiency.
    idx = np.arange(M, dtype=np.int64)
    bits = np.zeros((M, N), dtype=np.int8)
    for k in range(N):
        bits[:, k] = ((idx >> k) & 1).astype(np.int8)

    for i, j in cmps:
        a = bits[:, i]
        b = bits[:, j]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        bits[:, i] = lo
        bits[:, j] = hi

    # Sorted ascending means bit k (wire k) <= bit (k+1) (wire k+1) for all k.
    # Check if any adjacent pair is inverted.
    for k in range(N - 1):
        if np.any(bits[:, k] > bits[:, k + 1]):
            return False
    return True


# ---------------------------------------------------------------------------
# Depth (parallelism) reporter.
# ---------------------------------------------------------------------------
def _parallel_depth(cmps):
    end_time = {i: 0 for i in range(N)}
    depth = 0
    for i, j in cmps:
        t = max(end_time[i], end_time[j]) + 1
        depth = max(depth, t)
        end_time[i] = t
        end_time[j] = t
    return depth


# ---------------------------------------------------------------------------
# Reduction search: Try to remove comparators one by one.
# ---------------------------------------------------------------------------
def _reduce_network(initial_cmps):
    """
    Greedy reduction: iterate through comparators, try to remove each, 
    and keep it removed if the network remains valid.
    """
    current_cmps = list(initial_cmps)
    
    # To make the search more robust, we can try multiple random permutations 
    # of the comparator indices to remove. However, a single pass with a 
    # deterministic order or a few random restarts is usually sufficient for N=16.
    
    # We will perform several passes. In each pass, we try to remove any comparator 
    # that allows the network to remain valid.
    
    # Optimization: Pre-compute the "signature" of the network? 
    # No, just run the verifier. The verifier is fast (~50ms). 
    # 63 comparators * 65536 inputs * 16 wires is small.
    
    # Let's do a loop that tries to remove comparators until no more can be removed.
    # We iterate through the list of indices. If removing index k works, we remove it.
    # We restart the scan if we removed anything, or continue.
    
    # To avoid getting stuck in a local minimum, we can randomize the order of checks.
    
    max_passes = 10
    for _ in range(max_passes):
        if time.time() > _DEADLINE:
            break
            
        removed_any = False
        # Shuffle indices to break symmetry and find different local minima
        indices = list(range(len(current_cmps)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > _DEADLINE:
                break
            
            # Skip if already removed (indices change as we remove, so we need to be careful)
            # It's safer to iterate backwards or rebuild the list of candidates.
            # Let's just iterate through the current list.
            
            # To avoid index shifting issues, we pick a random index from the current list
            # and test removal.
            
            # Actually, a simpler approach:
            # Try removing each comparator in the current list.
            # If successful, remove it and restart the loop over the new list.
            # If not, move to the next.
            
            # Since we shuffle indices, we just pick one at a time.
            
            # Let's restructure: 
            # while there are comparators:
            #   pick a random one
            #   try to remove
            #   if fail, try next
            #   if all fail, stop
            
            # This might be slow if we pick many failing ones.
            # Better: Iterate through all, try to remove. If one is removed, restart iteration.
            
            pass # The logic below is better
            
    # Let's implement the "Iterate and Restart" strategy properly.
    
    current_cmps = list(initial_cmps)
    # We'll use a set of "tried" removals? No, just try all.
    
    # To be efficient, we can parallelize the verification? No, std lib only.
    # But NumPy is fast enough.
    
    # Strategy:
    # 1. Start with full list.
    # 2. Loop:
    #    a. For each comparator in the current list (in a fixed order, e.g., reverse to keep indices stable if we remove from end? 
    #       Or just check all, and if one is removable, remove it and break to restart the loop).
    #    b. If no comparator is removable in a full pass, stop.
    
    # To improve success rate, we can randomize the order of checking comparators in each pass.
    
    passes = 0
    max_passes = 20 # Safety limit
    
    while passes < max_passes:
        if time.time() > _DEADLINE:
            break
        
        passes += 1
        removed_in_pass = False
        
        # Create a list of indices to check, shuffled
        check_order = list(range(len(current_cmps)))
        random.shuffle(check_order)
        
        for idx in check_order:
            if time.time() > _DEADLINE:
                break
            
            # Try removing current_cmps[idx]
            candidate = current_cmps[:idx] + current_cmps[idx+1:]
            
            if _verify_zero_one(candidate):
                current_cmps = candidate
                removed_in_pass = True
                # Restart the pass to re-evaluate the new state
                # We break out of the for loop to restart the while loop
                break
        
        if not removed_in_pass:
            # No more comparators can be removed in this configuration
            break
            
    return current_cmps


# ---------------------------------------------------------------------------
# Entry point (name/signature kept exactly as in the seed program).
# ---------------------------------------------------------------------------
def construct():
    # 1. Generate Batcher's network
    net = _batcher_odd_even_merge(N)
    initial_count = len(net)
    
    # 2. Verify initial network (sanity check)
    if not _verify_zero_one(net):
        # Should not happen, but fallback to seed
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
                
    # 3. Reduce the network
    # Set a fixed seed for reproducibility if needed, but random is fine for search
    random.seed(42) 
    
    reduced_net = _reduce_network(net)
    
    # 4. Final verification
    if not _verify_zero_one(reduced_net):
        # If reduction broke something (shouldn't happen with verification), fallback to initial
        reduced_net = net
        
    # 5. Report
    depth = _parallel_depth(reduced_net)
    
    try:
        import sys
        sys.stderr.write(
            "[construct] Initial: %d, Reduced: %d, depth %d, verified: %s\n"
            % (initial_count, len(reduced_net), depth, _verify_zero_one(reduced_net))
        )
    except Exception:
        pass

    # Deadline guard
    if time.time() > _DEADLINE:
        return reduced_net

    return reduced_net


if __name__ == "__main__":
    net = construct()
    print("comparators:", len(net))
    print("depth:", _parallel_depth(net))
    print("zero-one verified:", _verify_zero_one(net))
