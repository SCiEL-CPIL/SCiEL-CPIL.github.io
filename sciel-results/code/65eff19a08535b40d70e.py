import time
import math
from itertools import combinations

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a, b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 255
    # 255 = 3 * 5 * 17
    
    # Build the candidate pool: numbers in [1, 255] that share a factor with 255
    # i.e., divisible by 3, 5, or 17
    candidates = []
    for i in range(1, N + 1):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Ensure 255 is in the set
    if 255 not in candidates:
        candidates.append(255)
    
    # We need a subset S of candidates such that:
    # 1. 255 in S
    # 2. gcd(a, b) > 1 for all distinct a, b in S
    # 3. Maximize |S|
    
    # Since 255 = 3*5*17, any element in S must share a factor with 255 (otherwise gcd=1 with 255)
    # So S must be a subset of candidates.
    
    # Additionally, for any two elements a, b in S, they must share a common prime factor.
    # The prime factors involved are only 3, 5, 17 (since all elements divide some multiple related to 255's factors).
    # Wait, elements can have other prime factors too, but the gcd must be > 1.
    # Since each element shares at least one of {3,5,17} with 255, let's label each element by which of {3,5,17} divide it.
    
    # For two elements to have gcd > 1, they must share at least one prime factor.
    # If we restrict to elements that are ONLY divisible by primes in {3,5,17}, then the condition is simply
    # that no two elements have disjoint sets of prime factors from {3,5,17}.
    
    # However, an element like 9 = 3^2 only has factor 3. An element like 25 = 5^2 only has factor 5.
    # gcd(9, 25) = 1. So we cannot include both.
    
    # Let's classify each candidate by its "signature": which of {3, 5, 17} divide it.
    # There are 2^3 - 1 = 7 possible non-empty signatures.
    # Two elements are compatible if their signatures intersect (share at least one prime).
    
    # This is a maximum clique problem in a graph where nodes are candidates and edges exist if signatures intersect.
    # But actually, it's simpler: we want to select a set of signatures (with counts) such that all selected signatures are pairwise intersecting.
    # This is equivalent to finding a star in the hypergraph of signatures, or more precisely, 
    # the intersection of all selected signatures must be non-empty? No.
    # If we select signatures {3}, {5}, they don't intersect.
    # If we select {3,5}, {3,17}, {5,17}, they pairwise intersect but the triple intersection is empty.
    # However, gcd of elements with these signatures:
    # a in {3,5}, b in {3,17}, c in {5,17}.
    # gcd(a,b) >= 3, gcd(a,c) >= 5, gcd(b,c) >= 17. So they are pairwise compatible!
    # So we don't need a common prime for all. We just need pairwise intersection of signatures.
    
    # So the problem reduces to:
    # For each of the 7 signature types, we can include any number of elements with that signature.
    # The constraint is that the set of signature types we choose must be pairwise intersecting.
    # To maximize size, for a fixed set of compatible signature types, we include ALL elements of those types.
    
    # So we need to find the set of signature types T ⊆ {1..7} such that:
    # 1. For all s1, s2 in T, s1 ∩ s2 ≠ ∅
    # 2. Sum of counts of elements with signature in T is maximized.
    
    # Let's compute the count of elements for each signature type.
    # Signature is a bitmask of {3, 5, 17}.
    # Bit 0: 3, Bit 1: 5, Bit 2: 17
    
    counts = [0] * 8  # index 0..7, 0 is empty (not used)
    
    for num in candidates:
        sig = 0
        if num % 3 == 0: sig |= 1
        if num % 5 == 0: sig |= 2
        if num % 17 == 0: sig |= 4
        counts[sig] += 1
    
    # Now we need to find a subset of {1..7} that is pairwise intersecting and maximizes sum of counts.
    # There are 2^7 = 128 possible subsets. We can brute force all.
    
    best_size = 0
    best_sig_set = []
    
    for mask in range(1, 128):
        # Check if the set of signatures in mask is pairwise intersecting
        sigs = [i for i in range(1, 8) if (mask & (1 << i))]
        valid = True
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                if (sigs[i] & sigs[j]) == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            size = sum(counts[s] for s in sigs)
            if size > best_size:
                best_size = size
                best_sig_set = sigs
    
    # Construct the set
    result = []
    for num in candidates:
        sig = 0
        if num % 3 == 0: sig |= 1
        if num % 5 == 0: sig |= 2
        if num % 17 == 0: sig |= 4
        if sig in best_sig_set:
            result.append(num)
    
    # Ensure 255 is included. 255 has signature 1|2|4 = 7.
    # Is 7 in best_sig_set?
    # If not, we must force it? No, the problem requires 255 in the set.
    # If the optimal set of signatures doesn't include 7, we must adjust.
    # But wait, if we don't include 7, we can't include 255.
    # So we must restrict our search to subsets T that include 7?
    # Or simply, if 7 is not in best_sig_set, we need to find the best set that DOES include 7.
    
    if 7 not in best_sig_set:
        best_size = 0
        best_sig_set = []
        for mask in range(1, 128):
            if not (mask & (1 << 7)): # Must include signature 7
                continue
            sigs = [i for i in range(1, 8) if (mask & (1 << i))]
            valid = True
            for i in range(len(sigs)):
                for j in range(i + 1, len(sigs)):
                    if (sigs[i] & sigs[j]) == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                size = sum(counts[s] for s in sigs)
                if size > best_size:
                    best_size = size
                    best_sig_set = sigs
        
        result = []
        for num in candidates:
            sig = 0
            if num % 3 == 0: sig |= 1
            if num % 5 == 0: sig |= 2
            if num % 17 == 0: sig |= 4
            if sig in best_sig_set:
                result.append(num)
    
    # Verify 255 is in result
    if 255 not in result:
        result.append(255)
        # This might break the gcd condition if 255 is not compatible with others?
        # But 255 has signature 7, which intersects with all non-empty signatures.
        # So adding 255 to any valid set of signatures is always compatible.
    
    return result
