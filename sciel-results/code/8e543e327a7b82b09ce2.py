import numpy as np
import time

N = 16

# EVOLVE-BLOCK-START

def _correct_bitonic_merge(n):
    """
    Correct Batcher Bitonic Merge for n wires (n must be power of 2).
    Generates n-1 comparators.
    """
    comps = []
    k = n // 2
    while k >= 1:
        for i in range(n - k):
            comps.append((i, i + k))
        k //= 2
    return comps

def _correct_bitonic_sort(n):
    """
    Correct Batcher Bitonic Sort for n wires.
    """
    if n == 1:
        return []
    
    half = n // 2
    
    # Left half: Ascending
    left = _correct_bitonic_sort(half)
    
    # Right half: Descending
    # Generate ascending, then reverse indices
    right_asc = _correct_bitonic_sort(half)
    right_desc = []
    for i, j in right_asc:
        i_new = half - 1 - j
        j_new = half - 1 - i
        right_desc.append((half + i_new, half + j_new))
        
    # Merge
    merge = _correct_bitonic_merge(n)
    
    return left + right_desc + merge

def _merge(k):
    """
    Generates comparators to merge two sorted lists of length k.
    The lists are assumed to be at indices 0..k-1 and k..2k-1.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    # 1. Merge even-indexed elements
    # Virtual indices 0..k_half-1 map to 0, 2, ..., k-2
    # Virtual indices k_half..k-1 map to k, k+2, ..., 2k-2
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    # 2. Merge odd-indexed elements
    # Virtual indices 0..k_half-1 map to 1, 3, ..., k-1
    # Virtual indices k_half..k-1 map to k+1, k+3, ..., 2k-1
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
        
    # 3. Final comparisons between adjacent pairs (2i+1, 2i+2)
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    
    return even_comps + odd_comps + final_comps

def _batcher(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    """
    if n == 1:
        return []
    
    k = n // 2
    # Sort the two halves
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    
    # Merge the two sorted halves
    merge = _merge(k)
    
    return left + right + merge

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate the correct Batcher Bitonic Sort network
    # This should be 49 comparators for N=16
    net = _correct_bitonic_sort(N)
    
    # Also generate Batcher Odd-Even Merge Sort (63 comps) as a secondary seed?
    # No, 49 is better. Let's stick with 49.
    
    # Precompute inputs for validity check
    # 2^16 = 65536 inputs.
    # We can represent these as integers 0..65535 where bit i represents wire i.
    # Wire i is 1 if bit i is set.
    # Sorting means the bits are in non-decreasing order (0s then 1s).
    # So for a sorted network, the output must be of the form:
    # If sum(bits) = k, then output has k ones at the end (highest indices) and 0s at the start.
    # i.e., output integer must be (1 << N) - (1 << k) ? No.
    # If k ones, they should be at indices N-k .. N-1.
    # Value = sum(1 << i for i in range(N-k, N)) = (1 << N) - (1 << (N-k)).
    # Example: N=4, k=1 (one 1). Sorted: 0001 (value 1). Formula: 16 - 8 = 8 (1000). Wait.
    # Min to wire i, Max to wire j.
    # If we sort ascending, smallest values go to low indices.
    # So 0s go to low indices, 1s go to high indices.
    # Input: 0011 (3). Sorted: 0011.
    # Input: 1100 (12). Sorted: 0011 (3).
    # Input: 1010 (10). Sorted: 0011 (3).
    # So if there are k ones, the output must have ones at positions N-k to N-1.
    # Value = sum(2^i for i in N-k..N-1) = 2^(N-1) + ... + 2^(N-k).
    # This is equal to (2^N - 2^(N-k)).
    
    # Let's precompute the target value for each possible count of 1s.
    # But we need to check if the network sorts EVERY input.
    # We can simulate all 65536 inputs.
    # N=16, 65536 inputs.
    # Simulation:
    # wires is a 16-bit integer.
    # Comparator (i, j):
    # bit_i = (wires >> i) & 1
    # bit_j = (wires >> j) & 1
    # min_val = bit_i & bit_j
    # max_val = bit_i | bit_j
    # wires = (wires & ~(1 << i)) | (min_val << i)
    # wires = (wires & ~(1 << j)) | (max_val << j)
    
    # This is fast in Python if we use integers?
    # 65536 inputs * ~50 comparators = 3.2M operations.
    # Python loops might be slow.
    # Let's use NumPy vectorization.
    
    # inputs: array of shape (65536, 16) of 0s and 1s.
    # Or just process in batches.
    
    # Let's try to do it with integers in a loop first, maybe it's fast enough?
    # 65536 * 50 = 3.2M. In Python, 3.2M simple ops is ~0.1s.
    # So a full check is very fast.
    
    def is_valid(network):
        # inputs are 0..65535
        # For each input, simulate and check if output is sorted.
        # Sorted condition: output == target[popcount(input)]
        
        # Precompute popcounts and targets?
        # popcount of x is x.bit_count() in Python 3.10+
        # Target for k ones: (1 << 16) - (1 << (16 - k))
        
        for x in range(1 << N):
            w = x
            for i, j in network:
                bi = (w >> i) & 1
                bj = (w >> j) & 1
                min_v = bi & bj
                max_v = bi | bj
                w = (w & ~(1 << i)) | (min_v << i)
                w = (w & ~(1 << j)) | (max_v << j)
            
            k = x.bit_count()
            target = (1 << N) - (1 << (N - k))
            if w != target:
                return False
        return True
    
    best_net = net
    best_count = len(best_net)
    
    # Verify initial network
    if not is_valid(best_net):
        # Fallback
        best_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        best_count = len(best_net)
    
    # Local Search: Try to remove comparators
    while time.time() < deadline:
        improved = False
        indices = list(range(len(best_net)))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            if idx >= len(best_net):
                continue
                
            candidate = best_net[:idx] + best_net[idx+1:]
            
            if is_valid(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break # Restart with new list
                
        if not improved:
            # Try a few swaps to escape local minima?
            swap_improved = False
            for _ in range(10):
                if time.time() > deadline:
                    break
                if len(best_net) < 2:
                    break
                i, j = np.random.choice(len(best_net), 2, replace=False)
                candidate = best_net[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                
                if is_valid(candidate):
                    best_net = candidate
                    swap_improved = True
                    break
                    
            if not swap_improved:
                break # Stuck
                
    return best_net

# EVOLVE-BLOCK-END
