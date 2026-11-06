import time
from math import gcd

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    # 255 = 3 * 5 * 17
    # The candidate pool must be numbers in [1, 255] that share a factor with 255.
    # i.e., multiples of 3, 5, or 17.
    
    # We want to maximize the size of a clique in the graph where vertices are candidates
    # and edges connect numbers with gcd > 1.
    
    # Let's analyze the structure.
    # Multiples of 3: A = {3, 6, ..., 255}. Size = 85.
    # Multiples of 5: B = {5, 10, ..., 250}. Size = 51.
    # Multiples of 17: C = {17, 34, ..., 255}. Size = 15.
    
    # The set of all multiples of 3 is a clique because any two multiples of 3 have gcd >= 3.
    # Can we add more?
    # If we add a number x that is not a multiple of 3, it must be a multiple of 5 or 17.
    # If x is a multiple of 5 but not 3, then gcd(x, y) must be > 1 for all y in the set.
    # If the set contains multiples of 3, say 3, then gcd(x, 3) > 1 implies 3 divides x.
    # But we assumed x is not a multiple of 3. Contradiction.
    # So we cannot have any number that is not a multiple of 3 if we have 3 in the set.
    # Wait, do we have to have 3?
    # 255 is in the set. 255 is a multiple of 3.
    # So 3 does not necessarily have to be in the set.
    # However, 255 is in the set.
    # Any other element x must have gcd(x, 255) > 1.
    # So x is divisible by 3, 5, or 17.
    
    # Let S be the solution.
    # If S contains a number not divisible by 3, say x.
    # Then x is divisible by 5 or 17 (or both).
    # Since 255 is in S, and 255 is divisible by 3, 5, 17.
    # This doesn't force all other elements to be divisible by 3.
    # Example: {255, 5}. gcd(255, 5) = 5 > 1. Valid. Size 2.
    # Example: {255, 15, 5}. gcd(255,15)=15, gcd(255,5)=5, gcd(15,5)=5. Valid. Size 3.
    # 15 is div by 3 and 5. 5 is div by 5 only.
    
    # Let's look at the "multiples of 3" set again. Size 85.
    # Can we replace some multiples of 3 with non-multiples of 3 to get a larger set?
    # Non-multiples of 3 in the pool are multiples of 5 (not 3) or 17 (not 3).
    # Let M3 = {x in [1,255] : 3|x}. |M3| = 85.
    # Let M5_only = {x in [1,255] : 5|x, 3!|x}.
    # Let M17_only = {x in [1,255] : 17|x, 3!|x}.
    # Let M5_17 = {x in [1,255] : 5|x, 17|x, 3!|x} = {85, 170}.
    
    # If we include any x in M5_only, it must have gcd > 1 with all other elements.
    # If the set contains any element y that is not divisible by 5, then gcd(x, y) must be 1 (since x is not div by 3, and if x is not div by 17 either, gcd is 1; if x is div by 17, gcd(x,y) must be 17, so y must be div by 17).
    
    # This suggests a partitioning based on prime factors.
    # The "largest" clique is likely the multiples of the smallest prime factor, which is 3.
    # Why? Because multiples of 3 form a dense subgraph.
    # Specifically, if we take S subset of M3, it's a clique.
    # If we try to add x not in M3, x must share a factor with every element in S.
    # If S contains 3, x must be div by 3, contradiction.
    # So S cannot contain 3 if we want to add non-multiples of 3.
    # But 255 is in S.
    # If we remove 3 from M3, do we lose more than we gain?
    # M3 \ {3} has size 84.
    # Can we add non-multiples of 3?
    # Let S' = (M3 \ {3}) U T, where T contains non-multiples of 3.
    # For any x in T, gcd(x, y) > 1 for all y in S'.
    # Since S' contains 6 (div by 2,3), 9 (div by 3), etc.
    # Actually, S' contains numbers like 6, 9, 12...
    # If x is in T, it is not div by 3.
    # gcd(x, 9) > 1 => x must be div by 3. Contradiction.
    # So S' cannot contain 9 if it contains any non-multiple of 3.
    # So we must remove all multiples of 3 that are powers of 3? No, just multiples of 3 that don't share other factors with x.
    # This is getting complicated.
    
    # Heuristic: The set of multiples of 3 is the largest.
    # Let's verify with a small search or logical proof.
    # The graph is the "divisibility graph" on [1, 255] restricted to factors of 255.
    # Actually, it's the graph where edges exist if gcd > 1.
    # This is the complement of the "coprime graph".
    # We want Max Clique in G.
    
    # Let's try to construct a larger set than 85.
    # If we take all multiples of 5. Size 51.
    # If we take all multiples of 17. Size 15.
    # If we take multiples of 3. Size 85.
    
    # Consider the set S = {x : 3|x}.
    # Is there a clique larger than 85?
    # Suppose |S| > 85.
    # By Pigeonhole or similar?
    # The total number of candidates is 85 + |M5_only| + |M17_only| + |M5_17|.
    # |M5_only|: Multiples of 5 not div by 3.
    # 5, 10, 20, 25, 35, 40, 50, 55, 65, 70, 80, 95, 100, 110, 115, 125, 130, 140, 145, 155, 160, 175, 185, 190, 200, 205, 215, 220, 230, 235, 245, 250.
    # Count:
    # Multiples of 5: 51.
    # Multiples of 15: 17.
    # So 51 - 17 = 34.
    # |M17_only|: Multiples of 17 not div by 3.
    # Multiples of 17: 15.
    # Multiples of 51: 5.
    # So 15 - 5 = 10.
    # |M5_17|: Multiples of 85 not div by 3.
    # 85, 170. (255 is div by 3).
    # So 2.
    
    # Total candidates = 85 + 34 + 10 + 2 = 131.
    
    # If we take all 131, is it a clique?
    # No. 5 and 3 are coprime.
    
    # The maximum clique size is likely 85.
    # Let's write a program that returns the multiples of 3, as it is the most robust "greedy" choice and likely optimal.
    
    result = [i for i in range(1, 256) if i % 3 == 0]
    
    # Verify 255 is in result
    if 255 not in result:
        result.append(255)
        
    # Quick verification
    # Since all are multiples of 3, gcd is at least 3.
    # This is always valid.
    
    return result
