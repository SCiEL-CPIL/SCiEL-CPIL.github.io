"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 18
NUM_BOOLS = 1 << N  # 262144

def _build_ooe_network(n):
    """Odd-even transposition network for n wires."""
    comps = []
    for phase in range(n):
        start = phase % 2
        for i in range(start, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _verify_bitmask(comps, n=N):
    """Verify network sorts all Boolean inputs using bitmask method.
    Returns True if correct."""
    # masks[i] = bitmask of inputs where wire i has value 1
    # For input k (0..2^n-1), wire i has value (k >> i) & 1
    # So masks[i] has bit k set iff (k >> i) & 1 == 1
    masks = np.zeros(n, dtype=np.uint64)
    # Actually we need 2^n bits, which for n=18 is 262144 bits = 4096 uint64s
    # Let's use a different approach: use Python integers or numpy arrays of uint64
    
    # Use numpy: each mask is an array of 2^n/64 = 4096 uint64 values
    num_words = (1 << n) // 64  # 4096 for n=18
    
    # Initialize masks: mask[i] has bit k set iff (k >> i) & 1
    masks = np.zeros((n, num_words), dtype=np.uint64)
    for i in range(n):
        # For each word w (0..num_words-1), bits 64*w .. 64*w+63
        # Bit k is set iff (k >> i) & 1 == 1
        # This creates a pattern: for i=0, alternating 1,0,1,0...
        # For i=1, 1,1,0,0,1,1,0,0...
        # We can compute this efficiently
        for w in range(num_words):
            base = 64 * w
            # Generate the 64-bit word for this position
            word = 0
            for bit in range(64):
                k = base + bit
                if (k >> i) & 1:
                    word |= (1 << bit)
            masks[i, w] = word
    
    # Apply comparators
    for (a, b) in comps:
        new_a = masks[a] & masks[b]
        new_b = masks[a] | masks[b]
        masks[a] = new_a
        masks[b] = new_b
    
    # Check: after sorting, wire 0 should be all 0s (min is 0 for all inputs except all-ones... 
    # wait, no. After sorting, wire 0 gets the minimum. For Boolean inputs, the minimum is 0
    # unless all inputs are 1. So wire 0 mask should be 0 (no input has 0 as minimum... 
    # actually wire 0 should have value 0 for all inputs except the all-1s input.
    # Hmm, let me think again.
    
    # Actually: after sorting, wire i should contain the (i+1)-th smallest value.
    # For Boolean inputs, the sorted result has (number of zeros) zeros followed by (number of ones) ones.
    # Wire 0: 0 unless all 1s. Wire 1: 0 unless all 1s or all but one are 1s, etc.
    
    # Easier check: the final masks should be "prefix" masks.
    # Specifically, mask[i] should be a set of consecutive bits at the high end.
    # mask[0] = 0 (only the all-1s input has wire 0 = 1, which is bit 2^n-1)
    # Wait: for the all-1s input (k = 2^n - 1), all wires are 1, so after sorting all wires are 1.
    # So mask[0] should have only bit (2^n - 1) set.
    # mask[1] should have bits (2^n - 2) and (2^n - 1) set, etc.
    
    # Actually the cleanest check: for each input k, the sorted output should have
    # the popcount(k) ones at the top. So wire i should be 1 iff i >= n - popcount(k).
    # Equivalently, mask[i] should contain exactly those k where popcount(k) > n - 1 - i,
    # i.e., popcount(k) >= n - i.
    
    # Let's just check: mask[i] should equal the set of k with popcount(k) >= n - i.
    # For i=0: popcount >= n, only k = 2^n-1. mask[0] = 1 << (2^n - 1)
    # For i=1: popcount >= n-1, k = 2^n-1 and all k with exactly one zero.
    
    # Simpler: check that masks are "monotone" in the right way.
    # mask[i] should be a suffix of bits: all bits from some position to 2^n-1.
    # And mask[i] should be a subset of mask[i+1].
    
    # Check monotonicity: mask[i] & ~mask[i+1] should be 0 for all i
    for i in range(n - 1):
        if np.any(masks[i] & ~masks[i + 1]):
            return False
    
    # Check the count: mask[0] should have exactly 1 bit set (the all-1s input)
    if np.sum(masks[0]) != 1:  # sum of uint64 doesn't count bits
        # Count bits in mask[0]
        bit_count = 0
        for w in masks[0]:
            bit_count += bin(w).count('1')
        if bit_count != 1:
            return False
    
    # Check mask[n-1] should have all bits set (every input has at least one 1? No!)
    # Wire n-1 is the maximum. For Boolean, max is 1 unless all zeros.
    # So mask[n-1] should have all bits except bit 0 (the all-zeros input).
    bit_count_last = 0
    for w in masks[n - 1]:
        bit_count_last += bin(w).count('1')
    if bit_count_last != (1 << n) - 1:
        return False
    
    return True

def _verify_bitmask_fast(comps, n=N):
    """Faster verification using precomputed popcount-based expected masks."""
    num_words = (1 << n) // 64  # 4096
    
    # Precompute expected masks: expected[i] = set of k where popcount(k) >= n - i
    # We'll build these once.
    # Actually, let's use a different verification: just check that the final state
    # matches the sorted order.
    
    # For each input k, sorted output has ones in wires [n-popcount(k), n-1]
    # So wire i = 1 iff popcount(k) >= n - i, i.e., popcount(k) + i >= n
    
    # Initialize
    masks = np.zeros((n, num_words), dtype=np.uint64)
    for i in range(n):
        for w in range(num_words):
            base = 64 * w
            word = 0
            for bit in range(64):
                k = base + bit
                if (k >> i) & 1:
                    word |= (1 << bit)
            masks[i, w] = word
    
    for (a, b) in comps:
        masks[a] &= masks[b]
        masks[b] = masks[a] | masks[b]
    
    # Check: for each wire i, mask[i] should be exactly {k : popcount(k) >= n - i}
    # Precompute expected bit counts for each wire
    # Wire i should have (number of k with popcount >= n-i) bits set
    # = sum_{j=n-i}^{n} C(n, j)
    
    from math import comb
    expected_counts = []
    for i in range(n):
        count = 0
        for j in range(n - i, n + 1):
            count += comb(n, j)
        expected_counts.append(count)
    
    for i in range(n):
        bit_count = 0
        for w in masks[i]:
            bit_count += bin(w).count('1')
        if bit_count != expected_counts[i]:
            return False
    
    return True

def _verify(comps, n=N):
    """Verify using bitmask method. Returns True if correct."""
    num_words = (1 << n) // 64
    masks = np.zeros((n, num_words), dtype=np.uint64)
    for i in range(n):
        for w in range(num_words):
            base = 64 * w
            word = 0
            for bit in range(64):
                k = base + bit
                if (k >> i) & 1:
                    word |= (1 << bit)
            masks[i, w] = word
    
    for (a, b) in comps:
        masks[a] &= masks[b]
        masks[b] = masks[a] | masks[b]
    
    # Check monotonicity and boundary conditions
    for i in range(n - 1):
        if np.any(masks[i] & ~masks[i + 1]):
            return False
    
    # Wire 0: only all-1s input
    bit_count_0 = sum(bin(w).count('1') for w in masks[0])
    if bit_count_0 != 1:
        return False
    
    # Wire n-1: all except all-zeros
    bit_count_last = sum(bin(w).count('1') for w in masks[n - 1])
    if bit_count_last != (1 << n) - 1:
        return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    n = N
    comps = _build_ooe_network(n)
    
    # Verify initial network
    if not _verify(comps, n):
        return comps
    
    best = list(comps)
    best_len = len(best)
    
    # Local search: try to remove comparators
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(comps)):
            if time.time() > deadline:
                break
            # Try removing comparator at idx
            trial = comps[:idx] + comps[idx+1:]
            if _verify(trial, n):
                comps = trial
                improved = True
                best = list(comps)
                best_len = len(comps)
                break  # Restart from beginning after a removal
    
    # Try swapping adjacent comparators to find new redundancies
    for _round in range(50):
        if time.time() > deadline:
            break
        found_swap = False
        for idx in range(len(comps) - 1):
            if time.time() > deadline:
                break
            # Swap comps[idx] and comps[idx+1]
            trial = comps[:]
            trial[idx], trial[idx+1] = comps[idx+1], comps[idx]
            if _verify(trial, n):
                # Check if this allows removing a comparator
                for j in range(len(trial)):
                    if time.time() > deadline:
                        break
                    trial2 = trial[:j] + trial[j+1:]
                    if _verify(trial2, n):
                        comps = trial2
                        best = list(comps)
                        best_len = len(comps)
                        found_swap = True
                        break
                if found_swap:
                    break
        if not found_swap:
            break
    
    return best
# EVOLVE-BLOCK-END
