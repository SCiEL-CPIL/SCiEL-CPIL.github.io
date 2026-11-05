import random

def construct():
    # We need a nonempty subset S of [1, 100] with no subset summing to 200, maximizing |S|.
    
    # Key observation: If we take a "tail" set {k, k+1, ..., 100}, we can check validity easily.
    # A subset sums to 200 if and only if there exist distinct elements in the set summing to 200.
    # Since all elements are positive, the smallest possible sum of 3 distinct elements in {k..100} is k + (k+1) + (k+2) = 3k + 3.
    # If 3k + 3 > 200, then no subset of size 3 or more can sum to 200.
    # Also, no single element is 200 (max is 100).
    # No two elements sum to 200 because the max sum of two distinct elements is 100 + 99 = 199.
    # Therefore, if 3k + 3 > 200, the set is valid.
    # 3k > 197 => k >= 66.
    # So {66, ..., 100} is valid. Size = 35.
    
    # Can we do better than 35?
    # Let's try to see if we can include some elements < 66.
    # If we include an element x < 66, we must avoid any subset of the remaining elements summing to 200 - x.
    # The remaining elements are mostly from the tail {66..100}.
    # The subset sums of {66..100} are very dense. In fact, every integer from 66 to 500 (sum of all) except a few small gaps?
    # Actually, the subset sums of consecutive integers are contiguous intervals.
    # The subset sums of {k, k+1, ..., m} form a contiguous range from 0 to sum(k..m) with some small initial gaps, but for large k, the range is very dense.
    # Specifically, for {66..100}, the minimum subset sum is 0, max is 500.
    # Does {66..100} contain a subset summing to 135? (200-65=135)
    # 66+69 = 135. Yes.
    # So if we include 65, we cannot include both 66 and 69.
    # To maximize size, we might swap elements.
    
    # However, there is a known result for this type of problem (subset sum avoidance).
    # The maximum size of a subset of [1, n] avoiding sum S is often related to the tail.
    # For S=200, n=100.
    # Let's check if size 36 is possible.
    # If we have 36 elements, by the pigeonhole principle or density arguments, is it forced to have a subset sum 200?
    # Consider the set {65, 67, 68, ..., 100} (removed 66, kept 65). Size 35.
    # 65 + 67 + 68 = 200. So this is invalid.
    # Consider {65, 66, 68, ..., 100} (removed 67, kept 65). Size 35.
    # 65 + 66 + 69 = 200. Invalid.
    # It seems hard to include 65 without removing multiple others.
    
    # Let's try a greedy construction to see if we can beat 35.
    # Start with [100, 99, ..., 1].
    # We want to keep as many as possible.
    # A known heuristic is to take the largest elements first, and only add a smaller element if it doesn't create a forbidden sum with the existing set.
    # But checking "creates a forbidden sum" is hard.
    
    # Alternative: The set {66, ..., 100} has size 35.
    # Is there a set of size 36?
    # If we take 36 numbers from [1, 100], the average value is at least (1+2+...+36)/36? No, we pick the largest to maximize sum, but we want to avoid 200.
    # Actually, to avoid small sums like 200, we want large numbers.
    # The 36 largest numbers are {65, ..., 100}.
    # We showed {65..100} is invalid (65+67+68=200).
    # Can we modify {65..100} to remove the sum 200 while keeping 36 elements?
    # We must remove at least one element from every triple summing to 200.
    # Triples in {65..100} summing to 200:
    # 65+67+68
    # 65+66+69
    # 66+67+67 (not distinct)
    # 64 is not in set.
    # Are there other triples?
    # 65+66+69, 65+67+68.
    # 66+67+67 no.
    # 66+66+68 no.
    # What about 4-element subsets?
    # 65+66+67+2? No 2 not in set.
    # Min 4-sum in {65..100} is 65+66+67+68 = 266 > 200.
    # So only 3-element subsets can sum to 200 in {65..100}.
    # The triples are:
    # (65, 67, 68)
    # (65, 66, 69)
    # (66, 67, 67) - invalid
    # (66, 66, 68) - invalid
    # (67, 67, 66) - invalid
    # Are there others?
    # Let a < b < c.
    # a=65: b+c=135. Pairs in {66..100} summing to 135: (66,69), (67,68).
    # a=66: b+c=134. Pairs in {67..100} summing to 134: (67,67) - not distinct. So no triples with min 66.
    # So only two triples: (65,67,68) and (65,66,69).
    # To break both, we can remove 65.
    # If we remove 65, we have {66..100}, size 35.
    # Can we remove something else to keep 65?
    # Remove 67 and 66? Then we remove 2 elements, size 34.
    # Remove 67 and 69? Size 34.
    # Remove 68 and 66? Size 34.
    # Remove 68 and 69? Size 34.
    # So removing 65 is the most efficient, yielding size 35.
    
    # Thus, it seems 35 is the maximum.
    
    # Let's double check if we can include an element < 65 while removing fewer than 1 from the tail?
    # If we include 64, we need to avoid sum 136 from the rest.
    # Rest is subset of {1..100} \ {64}.
    # If we keep {66..100}, does it sum to 136?
    # 66+70=136. Yes.
    # So (64, 66, 70) is a forbidden triple.
    # Also (64, 67, 69).
    # (64, 68, 68) no.
    # So including 64 creates more conflicts.
    
    # Therefore, the set {66, 67, ..., 100} is optimal with size 35.
    
    return list(range(66, 101))
