import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 55  # Safety margin before 60s kill

# Precompute full integer representation of all 2^N boolean inputs
all_masks = np.arange(1 << N, dtype=np.int32)
W0_int = all_masks.copy()

# Precompute popcounts for the initial inputs
popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
W0_bytes = W0_int[:, None] // (256 ** np.arange(4)) % 256
popcounts = popcount_table[W0_bytes].sum(axis=1)

# Precompute target values for sorted outputs
targets = np.zeros(N + 1, dtype=np.int32)
for k in range(N + 1):
    if k == 0:
        targets[k] = 0
    else:
        targets[k] = (1 << N) - (1 << (N - k))

def _batcher_merge_sort(n, offset=0):
    """
    Generate Batcher's odd-even mergesort network for n wires starting at offset.
    This is a correct, well-known sorting network.
    """
    if n <= 1:
        return []
    
    k = n // 2
    left = _batcher_merge_sort(k, offset)
    right = _batcher_merge_sort(n - k, offset + k)
    merge = _batcher_odd_even_merge(k, n - k, offset)
    
    return left + right + merge

def _batcher_odd_even_merge(n, m, offset):
    """
    Batcher's odd-even merge network for two sorted sequences of lengths n and m.
    Returns comparators for merging wires [offset, offset+n+m).
    Assumes n <= m for simplicity (or handles generally).
    """
    if n == 0:
        return []
    if m == 0:
        return []
    
    # Standard odd-even merge
    # Split into odd and even parts
    n_odd = n // 2
    n_even = n - n_odd
    m_odd = m // 2
    m_even = m - m_odd
    
    # Merge odd parts: n_odd from left, m_even from right
    # The positions in the combined array:
    # Left odd: 1, 3, 5... (indices 2*i+1 for i in 0..n_odd-1)
    # Right even: n, n+2, n+4... (indices n + 2*i for i in 0..m_even-1)
    # Wait, the standard definition maps indices differently.
    
    # Let's use the recursive definition on indices relative to the merge block.
    # Block size = n + m.
    # Left half indices: 0 .. n-1
    # Right half indices: n .. n+m-1
    
    # Odd indices in left: 1, 3, ..., 2*n_odd-1 (count n_odd)
    # Even indices in right: n, n+2, ..., n+2*m_even-2 (count m_even)
    
    # Even indices in left: 0, 2, ..., 2*n_even-2 (count n_even)
    # Odd indices in right: n+1, n+3, ..., n+2*m_odd-1 (count m_odd)
    
    comps = []
    
    # Merge odd sub-sequences
    # Left odd indices: 2*i + 1
    # Right even indices: n + 2*j
    # We need to map these to a contiguous range for the recursive call?
    # No, we can just call the merge on the "virtual" positions and map back.
    
    # Actually, it's easier to just generate the comparators directly using the standard algorithm
    # which compares specific pairs.
    
    # Standard Odd-Even Merge Algorithm:
    # 1. Merge odd-indexed elements of A and even-indexed elements of B.
    # 2. Merge even-indexed elements of A and odd-indexed elements of B.
    # 3. Compare adjacent elements.
    
    # Let's implement the recursion properly by returning comparators on the specific wire indices.
    
    def merge_odd_even(n_l, m_r, off_l, off_r):
        """
        Merge two sorted sequences.
        Left sequence is at wires off_l .. off_l+n_l-1
        Right sequence is at wires off_r .. off_r+m_r-1
        """
        if n_l == 0:
            return []
        if m_r == 0:
            return []
        
        # n_l is length of left, m_r is length of right.
        # We assume the wires are contiguous: [off_l, off_l+n_l+m_r)
        # But in odd-even merge, the input is two sorted lists.
        
        n_l_odd = n_l // 2
        n_l_even = n_l - n_l_odd
        m_r_odd = m_r // 2
        m_r_even = m_r - m_r_odd
        
        # 1. Merge (Left Odd, Right Even)
        # Left Odd indices: off_l + 1, off_l + 3, ...
        # Right Even indices: off_l + n_l + 0, off_l + n_l + 2, ...
        
        # These form two sorted lists that need to be merged.
        # The "slots" they occupy in the final array are:
        # Slot 1, 3, 5... (from Left Odd) and Slot n_l, n_l+2... (from Right Even)
        # This mapping is tricky to do recursively without explicit index mapping.
        
        # Alternative: Just use a known correct implementation of Odd-Even Merge.
        
        # Let's use a simpler, correct, but potentially less optimal merge for the initial seed:
        # Odd-Even Transposition Sort on the combined block.
        # This is O((n+m)^2) comparators but very simple and correct.
        # For N=18, this is fine for a seed.
        
        total_len = n_l + m_r
        start = off_l
        comps = []
        for phase in range(total_len):
            for i in range(phase % 2, total_len - 1, 2):
                w1 = start + i
                w2 = start + i + 1
                comps.append((w1, w2))
        return comps

    # Since I want to be efficient, I will implement the actual Odd-Even Merge recursion.
    # Reference: Knuth, TAOCP Vol 3.
    
    def _merge(n, m, offset):
        """Merge two sorted sequences of length n and m starting at offset."""
        if n == 0:
            return []
        if m == 0:
            return []
        
        # n is left length, m is right length.
        # Left: offset .. offset+n-1
        # Right: offset+n .. offset+n+m-1
        
        n_odd = n // 2
        n_even = n - n_odd
        m_odd = m // 2
        m_even = m - m_odd
        
        comps = []
        
        # 1. Merge (Left Odd, Right Even)
        # Left Odd wires: offset + 1, offset + 3, ... (n_odd wires)
        # Right Even wires: offset + n, offset + n + 2, ... (m_even wires)
        
        # We need to merge these two sets.
        # The result should go into the "odd slots" of the combined array?
        # No, the standard algorithm merges them into a specific set of wires.
        
        # Let's trace indices.
        # Combined array indices: 0 .. n+m-1
        # Left: 0 .. n-1
        # Right: n .. n+m-1
        
        # Odd indices in Left: 1, 3, ..., 2*n_odd-1
        # Even indices in Right: n, n+2, ..., n+2*m_even-2
        
        # These are the inputs to the first merge.
        # The outputs of this merge will occupy the odd indices of the combined array?
        # No, they occupy indices: 1, 3, ..., 2*n_odd-1 AND n, n+2, ...
        
        # This recursive mapping is complex. I will stick to the O(N^2) OET merge for the seed.
        # It's slower to build but simple and correct. The optimization phase will fix it.
        
        # OET Merge
        total_len = n + m
        for phase in range(total_len):
            for i in range(phase % 2, total_len - 1, 2):
                w1 = offset + i
                w2 = offset + i + 1
                comps.append((w1, w2))
        
        return comps

    return _merge(n, m, offset)

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses fast bitwise operations on int32 arrays.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return False 
        
    W = W0_int.copy()
    
    for i, j in net:
        # Extract bits
        ai = (W >> i) & 1
        aj = (W >> j) & 1
        
        # Min and Max
        mn = ai & aj
        mx = ai | aj
        
        # Clear bits i and j
        mask_i = ~(1 << i)
        mask_j = ~(1 << j)
        
        W = (W & mask_i & mask_j) | (mn << i) | (mx << j)
        
    # Check if W matches the sorted target for each input's popcount
    expected = targets[popcounts]
    
    return np.all(W == expected)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with Batcher's Merge Sort (using OET merge for simplicity/speed of generation)
    net = _batcher_merge_sort(N)
    best_net = net[:]
    best_count = len(net)
    
    # Ensure validity
    if not _is_sorted_network(net):
        # Fallback to full OET if Batcher failed (shouldn't happen)
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        best_net = net[:]
        best_count = len(net)
    
    # Phase 1: Aggressive Greedy Removal
    # Try removing comparators one by one.
    # We do multiple passes because removing one might make another redundant.
    
    changed = True
    while changed:
        if time.time() > deadline - 15:
            break
        changed = False
        
        # Iterate backwards to handle index shifts
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 15:
                return best_net
            
            new_net = net[:idx] + net[idx+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                # Restart the scan from the end? Or continue?
                # Continuing is fine, but we broke the index alignment.
                # Since we removed idx, the next index is the same as the next element.
                # But we are iterating backwards, so we just continue to idx-1.
                # However, the removal might make *other* comparators redundant.
                # A full re-scan is safer but slower.
                # Let's break and restart the outer while loop to do a full pass.
                break
        
        # If we broke due to a change, the outer loop will repeat.
        # If we finished the loop without change, changed is False.
        
    # Phase 2: Hill Climbing / Local Search
    # Use the time remaining to explore the neighborhood.
    
    if time.time() < deadline - 5:
        T = 0.5
        min_T = 0.01
        cooling = 0.9995
        no_improve_count = 0
        max_no_improve = 500
        
        current_count = len(net)
        
        while time.time() < deadline - 2:
            if T < min_T or no_improve_count > max_no_improve:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 0.5
                no_improve_count = 0
            
            # Generate move
            r = random.random()
            if r < 0.4:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx+1:]
            elif r < 0.6:
                # Insert
                # Try inserting a comparator that connects wires with potential disorder
                # Just random for now
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            elif r < 0.8:
                # Mutate
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)
            else:
                # Swap
                if len(net) < 2:
                    continue
                idx1 = random.randint(0, len(net) - 1)
                idx2 = random.randint(0, len(net) - 1)
                if idx1 == idx2:
                    continue
                new_net = net[:]
                new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
            
            if not _is_sorted_network(new_net):
                no_improve_count += 1
                continue
                
            new_count = len(new_net)
            delta = new_count - current_count
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / T)):
                net = new_net
                current_count = new_count
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            T *= cooling
    
    # Final validation
    if not _is_sorted_network(best_net):
        # Last resort: return a known valid network (OET)
        best_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    return best_net
