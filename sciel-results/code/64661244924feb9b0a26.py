"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def _check_dissociated_fast(selected):
    """
    Check if all reciprocal subset sums are distinct.
    Uses integer arithmetic by scaling with LCM to avoid floating point issues.
    For a set S, reciprocal subset sums are distinct iff for all non-empty subsets A, B of S,
    sum_{i in A} 1/i != sum_{j in B} 1/j.
    This is equivalent to: for all non-empty A, B with A != B, the sum is not zero.
    Actually, it's simpler: all 2^|S| subset sums must be distinct.
    We can check this by computing all subset sums as Fractions or scaled integers.
    
    For efficiency with small sets (max 22 elements), we use a set of tuples of (numerator, denominator)
    reduced by gcd, or we use a common denominator.
    
    A faster approach: 
    Two subset sums are equal iff their difference is zero.
    Let the elements be x_1, ..., x_k.
    Subset sum for mask m is sum_{i in m} 1/x_i.
    We need all these 2^k values to be distinct.
    
    We can compute them all and put in a set. With k <= 22, 2^22 = 4M which is too many to store in a set of Fractions.
    But we are maximizing cardinality. The seed has 9. Let's assume we won't go much above 12-15.
    2^15 = 32768, which is fine.
    2^16 = 65536, fine.
    2^17 = 131072, fine.
    2^18 = 262144, fine.
    2^19 = 524288, fine.
    2^20 = 1048576, might be slow with Fractions.
    
    To be safe and fast, we can use a dictionary of sums represented as (num, den) pairs reduced by gcd.
    Or, we can use a different check: 
    The set is dissociated if and only if the only integer linear combination sum c_i / x_i = 0 with c_i in {-1, 0, 1} is c_i = 0 for all i.
    This is the definition of dissociated sets in additive combinatorics.
    So we need to check that there is no non-trivial {-1, 0, 1} combination that sums to 0.
    This is equivalent to checking all 3^k combinations.
    For k=10, 3^10 = 59049.
    For k=15, 3^15 = 14M. This might be too slow in Python.
    
    Let's stick to the subset sum distinctness check but optimize it.
    We can use a set of hashes. But hash collisions are possible.
    Better to use exact arithmetic.
    
    Let's use the LCM approach.
    L = LCM(x_1, ..., x_k).
    Then each 1/x_i = L/x_i.
    Subset sums are integers: sum_{i in A} L/x_i.
    We just need all these integers to be distinct.
    L can be huge. For x_i up to 22, LCM is 232792560.
    This is fine for Python integers.
    
    So:
    1. Compute L = LCM of selected.
    2. Compute weights w_i = L // x_i.
    3. Generate all subset sums of w_i.
    4. Check if they are all distinct.
    """
    if not selected:
        return True
    L = 1
    for x in selected:
        L = L * x // gcd(L, x)
    weights = [L // x for x in selected]
    k = len(selected)
    # Generate all subset sums
    # There are 2^k subset sums.
    # We can use a set to check for duplicates.
    sums = set()
    # We can iterate through all masks
    for mask in range(1 << k):
        s = 0
        for i in range(k):
            if mask & (1 << i):
                s += weights[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _try_add(current_set, candidate, deadline):
    """Try adding candidate to current_set. Returns True if successful."""
    if time.time() > deadline:
        return False
    new_set = current_set + [candidate]
    new_set.sort()
    if _check_dissociated_fast(new_set):
        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Start with a known good set
    # The seed is [1, 2, 3, 5, 7, 11, 13, 17, 19]
    # Let's start with a smaller set and grow it.
    # Or start with the seed and try to improve?
    # The seed has 9 elements.
    # Let's try to find a set of size 10, 11, etc.
    
    # Strategy:
    # 1. Try to build a set greedily from small to large.
    # 2. Use random restarts with different orders.
    # 3. Try to add elements from [1, 22] in different orders.
    
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Try greedy construction with different orderings
    candidates = list(range(1, 23))
    
    # Try several random orders
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        current_set = []
        
        for c in candidates:
            if time.time() > deadline:
                break
            # Try to add c to current_set
            if _try_add(current_set, c, deadline):
                current_set.append(c)
                current_set.sort()
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            # If we reached a high size, we can stop early if we hit 22 (unlikely) or just continue
            # But let's continue to see if we can find even better
    
    # Also try starting from the seed and trying to add more
    # Or starting from subsets of the seed and adding others
    # But the greedy restarts should cover this.
    
    return best_set

# EVOLVE-BLOCK-END
