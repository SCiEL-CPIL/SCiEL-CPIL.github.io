import time
import numpy as np
from itertools import product

N = 16
MAX_COMPARATORS = 4096
TIME_LIMIT = 60  # seconds, with safety margin

def _batcher_oes(n):
    """
    Generate Batcher's Odd-Even Mergesort network for n inputs.
    Returns a list of comparators (i, j).
    """
    if n == 1:
        return []
    
    # Recursive step
    half = n // 2
    left = _batcher_oes(half)
    right = _batcher_oes(half)
    
    # Shift right comparators by half
    right = [(i + half, j + half) for i, j in right]
    
    # Merge phase
    merge = _batcher_merge(half)
    
    return left + right + merge

def _batcher_merge(k):
    """
    Generate the merge phase for two sorted lists of size k.
    Total wires: 2k.
    """
    comparators = []
    # Batcher's merge is a sequence of odd-even transposition passes
    # on the combined array, but only on the relevant wires.
    # Actually, Batcher's merge is defined recursively or via a specific pattern.
    # For k=1: (0,1)
    # For k=2: (0,2), (1,3), (0,1), (2,3) ? No.
    # Let's use the standard definition:
    # The merge of two sorted lists of length k uses:
    # 1. Odd-Even Transposition Sort on the 2k elements? No, that's not efficient enough.
    # Batcher's OES merge:
    # It consists of log2(k) stages.
    # Stage r (from 0 to log2(k)-1):
    #   For each pair of wires that are 2^r apart in the "merge pattern", compare.
    
    # A simpler way to generate Batcher's merge:
    # It is equivalent to an Odd-Even Transposition Sort on the 2k elements,
    # but restricted to the "merge" part?
    # Actually, Batcher's OES is:
    # 1. Sort left half.
    # 2. Sort right half.
    # 3. Merge:
    #    For r in 0..log2(k)-1:
    #       For i in 0..2k-1:
    #          If (i // 2^r) % 2 == 0:
    #             Compare (i, i + 2^r) if valid?
    
    # Let's use a known implementation of Batcher's OES.
    # Reference: https://en.wikipedia.org/wiki/Batcher%27s_odd%E2%80%93even_mergesort
    # The merge step for size 2k (from two k's) uses k * log2(k) comparators?
    # No, it uses 2k * log2(k) / 2?
    
    # Let's just use a recursive definition for the whole network which is easier to verify.
    # But I need the comparators.
    
    # Alternative: Use a pre-computed Batcher network for 16.
    # I will hardcode the Batcher network for 16 to be safe and fast.
    # Batcher OES for 16 has 62 comparators.
    
    # I will generate it recursively here to ensure correctness.
    # Merge(k) for two sorted lists of size k.
    # If k == 1: return [(0, 1)]
    # Else:
    #   Split the 2k wires into two halves of k? No.
    #   The merge step is:
    #   For r in range(log2(k)):
    #       step = 1 << r
    #       for i in range(0, 2*k, 2*step):
    #           for j in range(i, i + step):
    #               # Compare (j, j + step) and (j + step, j + 2*step)?
    #               # Actually, Batcher's merge is:
    #               # (j, j+step) and (j+step, j+2*step) is not quite right.
    
    # Let's use a simpler, known correct generator for Batcher's OES.
    # See: https://gist.github.com/...
    
    # I will use a hardcoded list for Batcher 16 to avoid implementation errors.
    # If I can't find it, I'll use Odd-Even Transposition (120) and prune.
    # But 120 is the seed. I need improvement.
    
    # Let's try to construct a 60-comparator network directly.
    # I will use a hardcoded optimal network.
    return []

# Since generating Batcher correctly is tricky without reference, I will use a 
# known optimal network hardcoded.
# Source: "Optimal Sorting Networks" or similar.
# A 60-comparator network for 16 inputs:

OPTIMAL_NETWORK = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)
]
# This list has 96 comparators? 8*12 = 96. Too many.

# I will use a fast search starting from a Batcher-like network.
# I'll generate a simple merge sort network which is ~70-80, then prune.

def _merge_sort_network(n):
    """Generate a simple merge sort network."""
    if n <= 1:
        return []
    half = n // 2
    left = _merge_sort_network(half)
    right = _merge_sort_network(half)
    right = [(i + half, j + half) for i, j in right]
    
    # Merge
    merge = []
    # Simple merge: compare all pairs (i, i+half) for i in 0..half-1
    # This is not a full merge, just a shuffle.
    # A proper merge requires more.
    # Let's use Odd-Even Transposition for the merge step?
    # OES on 2k elements takes 2k steps.
    # This is O(N^2).
    
    # I will use the hardcoded optimal network from a reliable source.
    # If I can't verify it, I'll use the seed.
    return left + right

# Given the risk of hardcoded errors, I will implement a fast local search 
# starting from the seed (120) and aggressively prune.
# The seed is 120. Optimal is 60.
# A fast greedy removal can likely get below 100 easily.

def _is_sorted_network_fast(network):
    """Fast verification using bit manipulation."""
    for mask in range(1 << N):
        vec = mask
        for i, j in network:
            vi = (vec >> i) & 1
            vj = (vec >> j) & 1
            if vi > vj:
                # Swap
                vec ^= (1 << i) | (1 << j)
        # Check if sorted: 0s then 1s
        # Count 1s
        count_ones = bin(vec).count('1')
        # The sorted vector should have count_ones 1s at the end
        # So the lower (N - count_ones) bits should be 0
        # And the upper count_ones bits should be 1
        expected = ((1 << count_ones) - 1) << (N - count_ones)
        if vec != expected:
            return False
    return True

def construct():
    """
    Return a fixed compare-exchange network that sorts all 16-input sequences.
    """
    import random
    start_time = time.time()
    deadline = start_time + 55  # 55s limit
    
    # Start with seed
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    best_network = network[:]
    
    # Greedy removal
    # Shuffle indices for randomization
    while time.time() < deadline:
        removed = False
        indices = list(range(len(best_network)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            # Try removing comparator at idx
            temp_network = best_network[:idx] + best_network[idx+1:]
            
            # Fast check
            if _is_sorted_network_fast(temp_network):
                best_network = temp_network
                removed = True
        
        if not removed:
            break
            
    return best_network

# EVOLVE-BLOCK-END
