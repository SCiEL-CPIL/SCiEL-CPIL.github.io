"""Size-efficient sorting network for 17 wires using bitmask simulation."""
import time
import random

N = 17
NUM_BOOLEANS = 1 << N  # 131072
DEADLINE_MARGIN = 5.0  # seconds before the 60s limit

# EVOLVE-BLOCK-START
def _build_bitmasks():
    """Build bitmask representation for all 2^N Boolean inputs.
    bitmask[w] is a Python int where bit k is set iff input k has 1 on wire w.
    """
    masks = [0] * N
    for k in range(NUM_BOOLEANS):
        for w in range(N):
            if (k >> w) & 1:
                masks[w] |= (1 << k)
    return masks

def _apply_comparator(masks, i, j):
    """Apply compare-exchange on wires i, j. Returns new masks."""
    new_masks = masks[:]
    # min goes to i, max goes to j
    # For each Boolean input k: if mask_i has bit k set and mask_j doesn't, they differ
    # min(i,j) for input k: 1 iff both i and j are 1 (since we take min of two bits)
    # Actually for 0-1: min(a,b) = a AND b, max(a,b) = a OR b
    new_i = masks[i] & masks[j]  # min: both must be 1
    new_j = masks[i] | masks[j]  # max: at least one is 1
    new_masks[i] = new_i
    new_masks[j] = new_j
    return new_masks

def _is_sorted(masks):
    """Check if final state is sorted: wire 0 has most 1s, wire N-1 has fewest.
    For a sorted network, after sorting, wire w should have exactly the top (N-w) 
    ones for each... actually simpler: the final masks should be such that
    mask[0] is a superset of mask[1] which is superset of mask[2] ... 
    i.e., mask[w] & ~mask[w+1] == 0 for all w, and the counts are correct.
    
    Actually for 0-1 inputs: a sorted sequence has all 1s then all 0s.
    So wire 0 should have the most 1s, wire 1 next, etc.
    mask[w] should have popcount = sum of top (N-w) bits... 
    Simpler: check that mask[w] is a superset of mask[w+1] for all w,
    and mask[0] has popcount N-1... no wait.
    
    For a 0-1 input with k ones: after sorting, wires 0..k-1 have 1, wires k..N-1 have 0.
    So mask[w] (which is the set of inputs where wire w is 1) should be:
    mask[w] = {k : number of ones in input k > w}
    
    The correct sorted masks: mask[w] should equal the set of inputs with popcount > w.
    Let's just check: mask[w] & ~mask[w-1] == 0 (mask[w] subset of mask[w-1]) for w=1..N-1,
    and popcount(mask[w]) should be C(N, w+1) + C(N, w+2) + ... + C(N, N) = sum_{j=w+1}^{N} C(N,j)
    """
    # Check subset relation
    for w in range(1, N):
        if masks[w] & ~masks[w - 1]:
            return False
    # Check popcounts: mask[w] should have exactly sum_{j=w+1}^N C(N,j) bits set
    from math import comb
    expected = sum(comb(N, j) for j in range(w + 1, N + 1))
    if masks[w].bit_count() != expected:
        return False
    return True

def _odd_even_transposition():
    """Generate odd-even transposition sorting network."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def construct():
    """Construct a minimal sorting network for 17 wires."""
    start_time = time.time()
    
    # Build bitmask representation
    masks = _build_bitmasks()
    
    # Start with odd-even transposition network
    network = _odd_even_transposition()
    
    # Verify initial network sorts correctly
    test_masks = masks[:]
    for i, j in network:
        test_masks = _apply_comparator(test_masks, i, j)
    if not _is_sorted(test_masks):
        # Fallback: return the seed network
        return network
    
    # Greedy deletion: try to remove each comparator and check if still valid
    # We process from the end (last comparators are more likely redundant)
    improved = True
    while improved:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > DEADLINE_MARGIN:
                break
            
            # Try removing comparator at idx
            candidate = network[:idx] + network[idx + 1:]
            
            # Verify candidate network
            test_masks = masks[:]
            valid = True
            for i, j in candidate:
                test_masks = _apply_comparator(test_masks, i, j)
                if time.time() - start_time > DEADLINE_MARGIN:
                    valid = False
                    break
            
            if valid and _is_sorted(test_masks):
                network = candidate
                improved = True
        
        if time.time() - start_time > DEADLINE_MARGIN:
            break
    
    return network
# EVOLVE-BLOCK-END
