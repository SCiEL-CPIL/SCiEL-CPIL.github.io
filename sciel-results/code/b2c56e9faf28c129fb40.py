"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s out of 90s wall-clock limit
    
    N = 255
    
    # The current seed returns multiples of 5 in [1,255], which is 51 elements.
    # We can do much better by using multiples of 3 (85 elements) or a mix.
    # Multiples of 3: 3, 6, 9, ..., 255 -> 85 elements. All pairs share factor 3.
    # Multiples of 5: 5, 10, ..., 255 -> 51 elements. All pairs share factor 5.
    # 
    # Can we do better than 85? 
    # The key insight: if we take all multiples of 3, we get 85 elements.
    # Can we add any non-multiple of 3? Only if it shares a factor > 1 with ALL multiples of 3.
    # A number x not divisible by 3 must share a factor with every multiple of 3.
    # 3 itself only shares factor with multiples of 3. So x must share a factor with 3, 
    # meaning x must be divisible by 3. Contradiction. So we can't add non-multiples of 3.
    #
    # What about a different base? Multiples of 2: 2,4,...,254 -> 127 elements. 
    # But 255 is odd, so 255 is not in this set. We need 255 in the set.
    #
    # Since 255 = 3 * 5 * 17, any element in our set must share a factor with 255,
    # i.e., must be divisible by 3, 5, or 17.
    #
    # Let's try: all numbers in [1,255] divisible by 3 OR 5 OR 17.
    # But we need gcd(a,b) > 1 for ALL pairs. This is much harder.
    #
    # The simplest valid set containing 255 with maximum size is likely all multiples of 3
    # that are <= 255, since 255 is divisible by 3. That gives 85 elements.
    #
    # Alternatively, all multiples of 5: 51 elements (smaller).
    # All multiples of 17: 15 elements (smaller).
    #
    # Can we combine? E.g., multiples of 3 AND multiples of 5? 
    # No - we need a single set where every pair has gcd > 1.
    # If we take {multiples of 3} ∪ {multiples of 5}, then 3 and 5 are in the set, 
    # gcd(3,5)=1. Invalid.
    #
    # What about a set where every element is divisible by 3 OR every element is divisible 
    # by 5? We can't mix. So the best single-prime approach is multiples of 3 (85 elements).
    #
    # But wait - can we use a composite requirement? E.g., every element divisible by 3 
    # or 15? That's just multiples of 3.
    #
    # What about: every element shares a common factor with 255, AND the set is "clique" 
    # in the non-coprime graph? The largest clique containing 255.
    #
    # Let's verify: multiples of 3 up to 255. Count = 255//3 = 85.
    # All pairs have gcd >= 3 > 1. 255 is included. Valid.
    
    # Let's also check if we can get more than 85 by being clever.
    # Consider numbers that are multiples of 3: 85 numbers.
    # Is there a larger set? 
    # 
    # The total numbers in [1,255] divisible by 3, 5, or 17:
    # |A3 ∪ A5 ∪ A17| where Ak = multiples of k
    # |A3| = 85, |A5| = 51, |A17| = 15
    # |A3∩A5| = 255//15 = 17
    # |A3∩A17| = 255//51 = 5
    # |A5∩A17| = 255//85 = 3
    # |A3∩A5∩A17| = 255//255 = 1
    # Union = 85+51+17-17-5-3+1 = 129
    #
    # But we can't just take all 129. We need pairwise gcd > 1.
    # The multiples of 3 alone give 85. Can we add some from A5 or A17?
    # 
    # If we add a number x divisible by 5 but not 3, it must share a factor with every 
    # multiple of 3 in our set. Take 3: gcd(x,3) must be > 1, so 3|x. But x not divisible 
    # by 3. Contradiction. So we can't add any non-multiple of 3 if 3 is in the set.
    #
    # What if we remove 3 from our multiples-of-3 set? Then we could potentially add 
    # numbers divisible by 5. But we'd lose 1 element and gain at most... let's see.
    # 
    # This becomes a maximum clique problem. For now, 85 is a solid baseline.
    # Let's return multiples of 3.
    
    result = list(range(3, 256, 3))
    # Verify 255 is in the set
    if 255 not in result:
        result.append(255)
    
    return result
# EVOLVE-BLOCK-END
