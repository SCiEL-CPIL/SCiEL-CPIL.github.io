import time
import numpy as np

N = 18
SEED = 153  # Odd-even transposition network size

def _odd_even_transposition(n):
    """Generate the standard odd-even transposition sorting network."""
    comps = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                comps.append((i, i + 1))
    return comps

def _bitonic_sort_network(n):
    """Generate a bitonic sorting network for n wires (pads to next power of 2)."""
    # Find next power of 2
    m = 1
    while m < n:
        m *= 2
    
    comps = []
    _build_bitonic(comps, 0, m, 1)
    # Filter out comparators that touch wires >= n
    # Actually, for bitonic sort on m wires, all wires 0..m-1 are used.
    # We need to map this back to n wires.
    # The bitonic network sorts m elements. If we only care about n < m,
    # we can just take the network and ignore wires >= n?
    # No, the comparators involve wires >= n.
    # We need a bitonic network that works on exactly n wires?
    # Bitonic sort requires power of 2.
    # Alternative: Use a known small network or stick with OET + pruning.
    # Let's stick with OET for simplicity and correctness, but use a better initial network if possible.
    # For now, return OET if n is not power of 2, or a proper bitonic if it is.
    if m == n:
        return comps
    else:
        # Fallback to OET
        return _odd_even_transposition(n)

def _build_bitonic(comps, lo, hi, direction):
    """Recursive helper for bitonic sort. direction: 1 for ascending, -1 for descending."""
    length = hi - lo
    if length <= 1:
        return
    
    half = length // 2
    _build_bitonic(comps, lo, lo + half, 1)
    _build_bitonic(comps, lo + half, hi, -1)
    
    # Merge phase
    for i in range(1, half + 1):
        for j in range(half):
            a = lo + j
            b = lo + j + i
            if direction == 1:
                comps.append((a, b))
            else:
                comps.append((b, a)) # Ensure i < j for the pair format? 
                # The problem says 0 <= i < j < N.
                # If we swap, we need to ensure the first element is smaller index.
                # The comparator (x, y) puts min at x and max at y.
                # If we want descending order in this sub-array, we want max at smaller index.
                # So we compare (a, b) but put max at a? No, the interface is fixed: min to i, max to j.
                # So to get descending, we need to use comparators that effectively sort in reverse?
                # Actually, bitonic sort works by having bitonic sequences.
                # Let's just generate a standard bitonic sort for ascending and assume it works.
                # For simplicity and correctness, let's just use OET. It's safer.

def _verify_network(net, time_limit=None, start_time=None):
    """
    Verify if the network sorts all 2^N binary inputs using the 0-1 principle.
    Uses NumPy for vectorized simulation.
    """
    if start_time is None:
        start_time = time.time()
    
    if not net:
        return N <= 1
    
    # Precompute bit positions for each comparator
    # Wire i corresponds to bit (N - 1 - i) in the integer mask
    # Let's stick to the convention: wire 0 is MSB.
    # Bit index for wire w is N - 1 - w.
    
    # We will simulate all 2^N inputs at once.
    # State is a 2D array of shape (2^N, N), where state[k, w] is the bit at wire w for input k.
    # Or we can use integer masks and bit operations.
    # Integer masks are faster in Python if we can vectorize?
    # NumPy doesn't handle bit operations on large integers well if they exceed 64 bits.
    # N=18, so masks fit in 32-bit integers.
    
    masks = np.arange(1 << N, dtype=np.uint32)
    
    # For each comparator (i, j):
    # bi = N - 1 - i
    # bj = N - 1 - j
    # bit_i = (mask >> bi) & 1
    # bit_j = (mask >> bj) & 1
    # if bit_i > bit_j: mask ^= (1 << bi) | (1 << bj)
    
    # Vectorized:
    # bit_i = (masks >> bi) & 1
    # bit_j = (masks >> bj) & 1
    # swap_mask = (bit_i > bit_j) * ((1 << bi) | (1 << bj))
    # masks ^= swap_mask
    
    for i, j in net:
        bi = N - 1 - i
        bj = N - 1 - j
        
        bit_i = (masks >> bi) & 1
        bit_j = (masks >> bj) & 1
        
        # If bit_i > bit_j, swap
        # bit_i is 1, bit_j is 0 -> swap
        # bit_i is 0, bit_j is 1 -> no swap
        # bit_i is 1, bit_j is 1 -> no swap
        # bit_i is 0, bit_j is 0 -> no swap
        
        swap_mask = (bit_i & ~bit_j) * ((1 << bi) | (1 << bj))
        masks ^= swap_mask
        
        if time_limit is not None:
            if time.time() - start_time > time_limit:
                return None # Timeout/Incomplete

    # Check if sorted
    # Sorted means all 0s followed by all 1s.
    # For each mask, count the number of 1s.
    # The expected mask for k ones is (1 << k) - 1.
    
    # Count ones in each mask
    # np.bit_count is available in NumPy 1.17+? 
    # Let's use a lookup table or loop if needed.
    # For 2^18 = 262144 items, a loop is fine, but let's try vectorized.
    # We can use unpackbits or similar?
    # Simpler: 
    # sorted_mask = (1 << k) - 1
    # We need to check if mask == (1 << popcount(mask)) - 1
    
    # Calculate popcount for all masks
    # Use a trick or just loop since it's only 262k items
    # Actually, we can precompute popcounts for all 2^N masks once? 
    # No, we need it for the current state.
    
    # Let's check sortedness efficiently.
    # A mask is sorted if it has no '10' pattern.
    # i.e., mask & (mask >> 1) == 0? No, that's for no adjacent 1s.
    # Sorted (0...01...1) means once a 1 appears, all subsequent bits (towards LSB) are 1?
    # Wait, wire 0 is MSB.
    # Sorted: 0...01...1.
    # Example: N=4, 0011 (mask 3).
    # Bits: b3=0, b2=0, b1=1, b0=1.
    # Pattern: 0011.
    # If we shift right by 1: 0001.
    # mask & (mask >> 1) = 0011 & 0001 = 0001. Not 0.
    
    # Condition for sorted (ascending in value, 0s then 1s):
    # There is no '1' followed by a '0' to its right (lower significance).
    # i.e., no pair (b_k=1, b_{k-1}=0) for any k.
    # This is equivalent to: (mask & (mask >> 1)) == 0?
    # Let's test 0011 (3). 3 & 1 = 1. Not 0.
    # So that condition is for "no adjacent 1s".
    
    # Correct condition:
    # Let x = mask.
    # Let y = x >> 1.
    # We want to ensure that if a bit in x is 1, the bit to its right in x is also 1?
    # No, 0011: b1=1, b0=1. OK. b2=0, b1=1. OK.
    # If we have 0101 (5). b2=0, b1=1, b0=0. Not sorted.
    # Check: x & (x >> 1)? 5 & 2 = 0. 
    # So that doesn't work.
    
    # Let's just compute the expected mask and compare.
    # popcount = np.array([bin(m).count('1') for m in masks]) # Too slow in Python loop?
    # 262k iterations is ~0.1s in Python. Acceptable.
    
    # Better: Use NumPy bit count if available, or a lookup table for bytes.
    # Let's use a lookup table for 8-bit chunks.
    # masks are uint32.
    # popcount = (table[masks & 0xFF] + table[(masks >> 8) & 0xFF] + ...)
    
    table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    pc = (table[masks & 0xFF] + 
          table[(masks >> 8) & 0xFF] + 
          table[(masks >> 16) & 0xFF] + 
          table[(masks >> 24) & 0xFF])
    
    expected = (1 << pc) - 1
    
    if not np.all(masks == expected):
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    # Start with Odd-Even Transposition
    net = _odd_even_transposition(N)
    
    # Initial verification (should be fast)
    if not _verify_network(net):
        # Should not happen
        return net
    
    current_size = len(net)
    
    # Greedy Pruning with NumPy verification
    # We try to remove comparators one by one.
    # To make it more effective, we can try to remove comparators that are "redundant".
    # A simple heuristic: iterate through the network and try to remove each.
    
    # We will do multiple passes.
    # In each pass, we try to remove every comparator.
    # If removal is successful, we keep it removed and continue.
    # If not, we keep it.
    
    # To speed up, we can randomize the order or use a specific heuristic.
    # Let's just do sequential passes until no more can be removed.
    
    progress = True
    while progress and time.time() < deadline - 10.0:
        progress = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            
            candidate = net[:idx] + net[idx+1:]
            
            # Verify candidate
            # To save time, we can verify with a timeout?
            # No, verification is fast enough (~0.1s).
            # 153 comparators * 262144 inputs * 18 wires?
            # NumPy vectorized: 153 * (few ops on 262k array) ~ 153 * 0.001s = 0.15s.
            # So each check is ~0.15s.
            # 153 checks per pass = 23s.
            # This is too slow for multiple passes.
            
            # Optimization: 
            # We don't need to verify the whole network every time?
            # No, we do.
            
            # Can we speed up verification?
            # Maybe check a subset of inputs first?
            # If it fails on a subset, it's invalid.
            # If it passes a large random subset, it's likely valid.
            # But we need certainty.
            
            # Let's try to reduce the number of checks.
            # We can use a "witness" set of inputs that are known to be hard.
            # Or just accept the speed and do fewer passes.
            
            # Let's try one pass first.
            
            valid = _verify_network(candidate)
            
            if valid:
                net = candidate
                current_size = len(net)
                progress = True
                # Continue loop (index adjusted by range)
            
            # If invalid, keep the comparator.
        
        # If we made progress, another pass might find more redundancies.
        # But time is limited.
    
    # If time remains, try Simulated Annealing or Swap-based search?
    # Swap: Replace comparator (i,j) with (k,l).
    # This is a larger search space.
    
    # For now, return the pruned network.
    
    return net
