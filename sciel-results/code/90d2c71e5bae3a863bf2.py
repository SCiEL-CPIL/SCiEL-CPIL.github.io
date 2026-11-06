"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import product

N = 16

# EVOLVE-BLOCK-START
def _precompute_masks():
    """Precompute initial and target bitmasks for verification."""
    total_inputs = 1 << N
    all_mask = (1 << total_inputs) - 1
    
    # Initial ones[i]: bitmask of inputs where wire i (bit i of index) is 1
    init_ones = [0] * N
    for i in range(N):
        m = 0
        period = 1 << (i + 1)
        half = 1 << i
        # Bit i is 1 in the second half of each period
        for base in range(half, total_inputs, period):
            m |= ((1 << half) - 1) << base
        init_ones[i] = m
    
    # Target ones[i]: bitmask of inputs where wire i in sorted output is 1
    # Sorted output: 0s then 1s.
    # Wire i is 1 if number of ones in input >= N - i
    popcounts = [0] * total_inputs
    for x in range(total_inputs):
        # Efficient popcount using bin().count('1') is okay for 65536
        popcounts[x] = bin(x).count('1')
        
    target_ones = [0] * N
    for i in range(N):
        threshold = N - i
        m = 0
        for x in range(total_inputs):
            if popcounts[x] >= threshold:
                m |= (1 << x)
        target_ones[i] = m
        
    return init_ones, target_ones, all_mask


def _verify_network_fast(comparators, init_ones, target_ones, all_mask):
    """Verify sorting network using precomputed masks."""
    # Copy initial states
    # We only need to track 'ones' because 'zeros' is complement of 'ones'
    # But the update rule for zeros is different (OR vs AND).
    # zeros[a] = za | zb
    # ones[a] = oa & ob
    # Note: zeros[a] == all_mask ^ ones[a]
    # Let's check if we can derive zeros from ones.
    # za = ~oa, zb = ~ob
    # new_zeros[a] = ~oa | ~ob = ~(oa & ob) = ~new_ones[a].
    # Yes! So we only need to track `ones`.
    
    ones = init_ones[:]
    
    for (a, b) in comparators:
        oa, ob = ones[a], ones[b]
        # min goes to a, max goes to b
        # a gets 1 only if both were 1
        ones[a] = oa & ob
        # b gets 1 if either was 1
        ones[b] = oa | ob
        
    # Check if final ones match target ones
    for i in range(N):
        if ones[i] != target_ones[i]:
            return False
            
    return True


def construct():
    """Construct a sorting network with minimum comparators."""
    start_time = time.time()
    deadline = start_time + 50  # 50 second deadline (safe margin)
    
    # Precompute masks
    init_ones, target_ones, all_mask = _precompute_masks()
    
    rng = random.Random(42)
    
    # Start with odd-even transposition network
    # Phase p: compare (i, i+1) where i % 2 == p % 2
    best_network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best_count = len(best_network)
    
    # Verify initial network
    if not _verify_network_fast(best_network, init_ones, target_ones, all_mask):
        # Fallback to a known valid network if construction was wrong (should not happen)
        # But let's assume it's correct.
        pass

    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Local search: try removing comparators
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        current = best_network[:]
        
        # Try removing each comparator
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx + 1:]
            
            if _verify_network_fast(candidate, init_ones, target_ones, all_mask):
                best_network = candidate
                best_count = len(best_network)
                improved = True
                break  # Restart removal loop with new shorter list
        
        if not improved and time.time() < deadline - 5:
            # Try swapping comparators to escape local minima
            # Swapping doesn't reduce count, but might allow future removals
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                if len(best_network) < 2:
                    break
                i, j = rng.sample(range(len(best_network)), 2)
                candidate = best_network[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                
                # Optional: Verify swap? Swapping a valid network keeps it valid?
                # No, swapping comparators in a sorting network does NOT necessarily keep it valid.
                # So we must verify.
                if _verify_network_fast(candidate, init_ones, target_ones, all_mask):
                    best_network = candidate
                    # Don't set improved=True, just accept the swap and continue search
                    # Actually, if we accept a swap, we might enable removals.
                    # So we should break and re-enter the outer loop?
                    # Or just continue trying swaps?
                    # If we find a valid swap, we adopt it. Then we try removals again.
                    break

    return best_network
# EVOLVE-BLOCK-END
