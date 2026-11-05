"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

def construct():
    # We want a nonempty subset of [1, 255] containing 255,
    # with gcd(a,b) > 1 for all distinct pairs, maximizing cardinality.
    
    # 255 = 3 * 5 * 17
    # So any number in our set must share a factor with 255, i.e., be divisible by 3, 5, or 17.
    # But also, any two numbers in the set must have gcd > 1.
    
    # Let's think about this more carefully.
    # If we include 255, then every other element a must satisfy gcd(255, a) > 1.
    # So a must be divisible by 3, 5, or 17.
    
    # Now, if we have two numbers a and b both in the set, we need gcd(a, b) > 1.
    # This is a stronger condition.
    
    # Strategy: Consider subsets based on a common prime factor.
    # If all numbers are divisible by p, then any two have gcd >= p > 1.
    # So the set of all multiples of p in [1, 255] that also share a factor with 255 will work,
    # provided p divides 255 (so that 255 is included and shares factor p with all).
    
    # Wait, 255 must be in the set. 255 is divisible by 3, 5, 17.
    # If we take all multiples of 3 in [1,255], that includes 255 (since 255=3*85).
    # Any two multiples of 3 have gcd >= 3 > 1. So this works.
    # Size: floor(255/3) = 85.
    
    # Similarly, multiples of 5: floor(255/5) = 51. Includes 255.
    # Multiples of 17: floor(255/17) = 15. Includes 255.
    
    # Can we do better by combining?
    # Suppose we take numbers divisible by 3 OR 5 OR 17, but we need pairwise gcd > 1.
    # If we take a multiple of 3 (not 5 or 17) and a multiple of 5 (not 3 or 17), their gcd might be 1.
    # E.g., 3 and 5: gcd=1. So we can't just take the union.
    
    # What if we take all multiples of 3, plus some multiples of 5 that are also multiples of 3?
    # That's just multiples of 3 again.
    
    # Alternatively, take all multiples of 15 (3*5) plus 255? 255 is a multiple of 15? 255/15=17, yes.
    # Multiples of 15 in [1,255]: floor(255/15)=17. Any two multiples of 15 have gcd >= 15 > 1.
    # But this is smaller than 85.
    
    # What about a mixed approach? Let's think about the structure.
    # The key insight: if we partition the set into groups where each group has a common prime factor,
    # and ensure that any two elements from different groups still share a common factor, we can potentially do better.
    
    # Actually, let's reconsider. The condition is that for ALL distinct pairs, gcd > 1.
    # This is equivalent to saying the set is a clique in the graph where edges connect numbers with gcd > 1.
    
    # Let's try a greedy approach or analyze prime factors more carefully.
    
    # Key observation: If we pick a set S containing 255, then for every a in S, gcd(a, 255) > 1.
    # So a is divisible by 3, 5, or 17.
    
    # Now consider the prime factorization of each element. Each element must have at least one of {3, 5, 17} as a factor.
    
    # For two elements a and b in S, gcd(a,b) > 1 means they share at least one prime factor.
    
    # Let's try: take all multiples of 3. That's 85 elements. Can we add any element that's not a multiple of 3?
    # Such an element must be divisible by 5 or 17 (to share a factor with 255).
    # If we add an element divisible by 5 but not 3, it must share a factor with ALL multiples of 3 in our set.
    # But a multiple of 3 like 3*2=6 (if 6 is in our set) and an element divisible by 5 but not 3, e.g., 5: gcd(6,5)=1.
    # So we can't add 5 to the set of all multiples of 3.
    
    # What if we restrict our multiples of 3 to those that also share a factor with some candidate?
    # This is getting complex. Let's try a different approach.
    
    # Let's consider: what if all elements are divisible by a common prime p? Then we're fine.
    # The largest such set containing 255 is the set of all multiples of p where p divides 255.
    # p=3: 85 elements
    # p=5: 51 elements  
    # p=17: 15 elements
    
    # So 85 is a lower bound.
    
    # Can we exceed 85? Let's think...
    # Suppose we don't take ALL multiples of 3, but take a subset, and add some numbers not divisible by 3.
    # For the set to be valid, every pair must have gcd > 1.
    
    # Let's try: take all multiples of 3 that are also divisible by 5 or 17? No, that would be smaller.
    
    # Alternative: Consider that any element in S must be divisible by 3, 5, or 17.
    # Let S3 = elements divisible by 3, S5 = divisible by 5, S17 = divisible by 17.
    # S must be a subset of S3 ∪ S5 ∪ S17 containing 255.
    
    # For pairwise gcd > 1, if we have a ∈ S3\S5\S17 (divisible by 3 only among {3,5,17}) and b ∈ S5\S3\S17,
    # then gcd(a,b) must come from some other prime factor. But a might be 3*k where k is coprime to 5*17, and b=5*m where m is coprime to 3*17.
    # Then gcd(a,b) could be 1.
    
    # So to be safe, if we include an element that's "purely" divisible by one of {3,5,17} (and no other shared primes with other "pure" elements), we have constraints.
    
    # Let's try a computational approach: build the graph and find a large clique.
    # But 255 vertices is manageable for a heuristic.
    
    # The largest known pairwise non-coprime set containing 255 (3*5*17)
    # is the set of all multiples of the smallest prime factor, 3.
    # Size: 255 // 3 = 85.
    # Any two multiples of 3 have gcd >= 3 > 1.
    # 255 is a multiple of 3, so it is included.
    
    return [i for i in range(3, 256, 3)]
# EVOLVE-BLOCK-END
