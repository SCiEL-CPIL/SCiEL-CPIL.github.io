"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    """
    Construct a set of distinct integers in 1..256 such that:
    - Two subsets with different cardinalities have different sums.
    - Maximize the set size.
    
    Strategy:
    1. Use a bitwise DP to track achievable sums for each cardinality.
    2. Greedily add numbers from a structured lattice (powers of 2, then geometric, then others)
       if they don't cause a collision between different cardinalities.
    3. The key constraint: for any k1 != k2, the set of sums achievable with cardinality k1
       must be disjoint from the set of sums achievable with cardinality k2.
    
    We use bitsets: sums_by_k[k] is a bitset where bit s is 1 if sum s is achievable with cardinality k.
    When adding a new number x, we check:
      For each existing k, the new sums at cardinality k+1 are (sums_by_k[k] << x).
      We need to ensure that (sums_by_k[k] << x) does not intersect with sums_by_k[j] for any j != k+1.
      Also, (sums_by_k[k] << x) must not intersect with the current sums_by_k[k+1] (unless we're building it up).
    
    Actually, the condition is: for all k1 != k2, sums_by_k[k1] & sums_by_k[k2] == 0.
    
    When we add x, the new sums_by_k' are:
      sums_by_k'[0] = 1 (bit 0 set, empty subset)
      sums_by_k'[k] = sums_by_k[k] | (sums_by_k[k-1] << x) for k >= 1
    
    We need to verify that for all k1 != k2, sums_by_k'[k1] & sums_by_k'[k2] == 0.
    
    To make this efficient, we can check incrementally. But since the max size is 60 and max sum is 60*256 = 15360,
    the bitsets are manageable.
    
    Let's use a list of integers as bitsets. sums_by_k[k] is an int where bit s indicates sum s is achievable.
    
    We'll try numbers in a specific order to maximize the chance of adding many numbers.
    A good order might be: large numbers first, or numbers that are "well-separated".
    
    Actually, powers of 2 work perfectly: {1, 2, 4, 8, ..., 2^59} but 2^59 is way beyond 256.
    So we can only use {1, 2, 4, 8, 16, 32, 64, 128} which is 8 numbers.
    
    Let's think differently. We want to maximize the number of elements.
    
    The constraint is that subset sums of different cardinalities must be disjoint.
    
    Let S_k be the set of all subset sums of cardinality k.
    We need S_k ∩ S_j = ∅ for k ≠ j.
    
    This is a strong condition. Let's try to build the set greedily, trying numbers from 1 to 256,
    and for each number, check if adding it maintains the disjointness property.
    
    To check efficiently:
    - Maintain sums_by_k as a list of bitsets (Python ints).
    - When considering adding x:
      - Compute new_sums_by_k where new_sums_by_k[0] = 1, and for k >= 1:
        new_sums_by_k[k] = sums_by_k[k] | (sums_by_k[k-1] << x)
      - Check that for all k1 != k2, new_sums_by_k[k1] & new_sums_by_k[k2] == 0.
      - If valid, add x and update sums_by_k = new_sums_by_k.
    
    The maximum possible sum is 60 * 256 = 15360, so bitsets are 15361 bits, which is fine for Python ints.
    
    Let's try numbers in increasing order first, then if that doesn't give enough, try other orders.
    """
    
    max_val = 256
    max_size = 60
    
    # sums_by_k[k] is a bitset: bit s is 1 if sum s is achievable with cardinality k
    # Start with empty set: only sum 0 is achievable with cardinality 0
    sums_by_k = [0] * (max_size + 1)
    sums_by_k[0] = 1  # bit 0 set
    
    result = []
    
    # Try numbers from 1 to 256
    for x in range(1, max_val + 1):
        # Check if we can add x
        # Compute new_sums_by_k
        new_sums_by_k = [0] * (max_size + 1)
        new_sums_by_k[0] = 1
        
        valid = True
        for k in range(1, max_size + 1):
            new_sums_by_k[k] = sums_by_k[k] | (sums_by_k[k - 1] << x)
        
        # Check disjointness: for all k1 != k2, new_sums_by_k[k1] & new_sums_by_k[k2] == 0
        for k1 in range(max_size + 1):
            for k2 in range(k1 + 1, max_size + 1):
                if new_sums_by_k[k1] & new_sums_by_k[k2]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            result.append(x)
            sums_by_k = new_sums_by_k
            if len(result) >= max_size:
                break
    
    return result
# EVOLVE-BLOCK-END
