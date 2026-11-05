"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 17

# EVOLVE-BLOCK-START
def _count_sort_network(n):
    """
    Generate a sorting network using the bitonic sort algorithm.
    Bitonic sort is a well-known parallel sorting network with O(n log^2 n) comparators.
    For n=17, we pad to 32 (next power of 2) or use a recursive construction.
    However, bitonic sort on 17 wires isn't directly applicable since it requires power of 2.
    
    Alternative: Use the AKS sorting network? Too complex.
    
    Let's use a simpler approach: generate a sorting network via the "odd-even merge" or 
    use a known efficient construction.
    
    Actually, let's implement a recursive bitonic-like construction or use the 
    "brick wall" pattern which is known to be efficient.
    
    A better approach: Use the standard "bitonic sort" by padding to 32 wires, but that's wasteful.
    
    Let's try a different approach: Use the "0-1 principle" to verify, but for construction,
    let's use a known efficient network.
    
    The best known for 17 wires is 71 comparators. Let's try to construct one using 
    a combination of known sub-networks.
    
    Actually, let's implement a simple recursive sorting network using the 
    "merge sort" style which gives O(n log^2 n) comparators.
    
    For n=17, a merge-sort style network would have roughly:
    - Split into two halves, sort each, then merge.
    - The merge step for m elements requires m comparators in a specific pattern.
    
    Let's implement a recursive construction:
    1. If n <= 1, return empty.
    2. Split into left (n//2) and right (n - n//2).
    3. Recursively sort left and right.
    4. Merge the two sorted halves using a merge network.
    
    The merge network for two sorted sequences of length m and k can be done with 
    a specific pattern. For simplicity, let's use the "bitonic merge" if lengths are equal,
    or a more general merge.
    
    Actually, let's use a simpler and well-known construction: the "odd-even transposition"
    is already in the seed. Let's try "bitonic sort" by padding to 32 wires.
    
    Wait, padding to 32 wires would use more comparators. Let's calculate:
    Bitonic sort on 32 wires: 32 * 5 * 4 / 2 = 320 comparators? No, that's not right.
    
    Bitonic sort network for n = 2^k wires has (n/2) * k * (k+1)/2 comparators? 
    Actually, the number of comparators in a bitonic sort network for n wires is:
    (n/2) * sum_{i=1}^{log2(n)} i = (n/2) * (log2(n)(log2(n)+1)/2)
    
    For n=32: (32/2) * (5*6/2) = 16 * 15 = 240 comparators. That's worse than 136.
    
    Let's try a different approach. Let's use the "AKS" network? No, too complex.
    
    Let's try to construct a network using the "0-1 principle" and a greedy approach,
    but that's not guaranteed to be optimal.
    
    Actually, let's use a known efficient construction: the "brick wall" network.
    The brick wall network for n wires has approximately (n^2 / 4) * log2(n) comparators,
    which for n=17 is about (289/4) * 4.09 = 295, which is worse.
    
    Let's try to use a hybrid approach: use odd-even transposition for small n and 
    a more efficient method for larger n.
    
    Actually, let's just implement a correct sorting network using a recursive merge 
    approach, even if it's not optimal. The goal is to beat 136 comparators.
    
    Let's implement a simple recursive sorting network:
    - Sort left half
    - Sort right half  
    - Merge the two sorted halves
    
    The merge of two sorted sequences of lengths m and k can be done with a network 
    that has m + k - 1 comparators if we use a "selection network" style, but that's 
    not quite right.
    
    Actually, let's use the "bitonic merge" approach for equal-length halves.
    
    For now, let's implement a simple recursive construction and see how many 
    comparators we get.
    """
    if n <= 1:
        return []
    
    # Split into two halves
    left_size = n // 2
    right_size = n - left_size
    
    # Recursively sort left and right
    left_network = _count_sort_network(left_size)
    right_network = _count_sort_network(right_size)
    
    # Shift right network indices
    right_network = [(i + left_size, j + left_size) for i, j in right_network]
    
    # Merge the two sorted halves
    # Use a simple merge network: compare adjacent elements across the boundary
    # This is not optimal but should work
    merge_network = []
    for i in range(min(left_size, right_size)):
        merge_network.append((i, i + left_size))
    
    # We need to do multiple passes to fully merge
    # Let's do a simple "bubble" merge
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            merge_network.append((i, i + 1))
    
    return left_network + right_network + merge_network


def construct():
    """
    Construct a sorting network for 17 wires.
    Uses a recursive merge-sort style construction.
    """
    network = _count_sort_network(N)
    # Ensure we don't exceed the maximum
    if len(network) > 4096:
        # Fall back to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    return network
# EVOLVE-BLOCK-END
