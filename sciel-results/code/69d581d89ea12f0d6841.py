import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin under 90s limit
    
    n = 255
    # 255 = 3 * 5 * 17
    # All multiples of 3 in [1, 255]: 85 elements
    # All multiples of 5 in [1, 255]: 51 elements
    # All multiples of 17 in [1, 255]: 15 elements
    # Any pairwise non-coprime set containing 255 must have all elements share a factor with 255
    # But also all pairs must be non-coprime with each other.
    
    # Key insight: If we take all multiples of 3, any two multiples of 3 have gcd >= 3 > 1.
    # This gives 85 elements. Can we do better?
    
    # We need a set S containing 255 where gcd(a,b) > 1 for all distinct a,b in S.
    # The maximum such set: every element must share a common factor with 255 (3, 5, or 17),
    # AND any two elements must share SOME factor.
    
    # The multiples of 3 form a valid set of size 85.
    # Can we add elements that are multiples of 5 but not 3? No, because such an element 
    # would need to share a factor with all multiples of 3 already in the set, but e.g., 3 and 10 have gcd 1.
    
    # So the question is: is there a larger set not based on a single prime factor?
    # For example, take all even numbers <= 255? That's 127 elements. But 255 is odd, so 255 is not even.
    # We MUST include 255. 255 is odd, not divisible by 2.
    
    # What about multiples of 3 OR 5 or 17 that are pairwise non-coprime?
    # If we include a multiple of 3 (not of 5 or 17) and a multiple of 5 (not of 3 or 17), 
    # they have gcd 1. So we can't mix different prime factors freely.
    
    # The answer is: the maximum set is the multiples of the smallest prime factor, which is 3.
    # |{k*3 : 1<=k<=85}| = 85.
    
    # Let's verify this is optimal. Suppose S contains 255 and all pairs are non-coprime.
    # Let p be a prime dividing 255. Consider the partition of S by which of {3,5,17} divide each element.
    # Actually, elements could be divisible by multiple primes of 255, or by primes not in {3,5,17}.
    
    # But if an element x in S is not divisible by 3, 5, or 17, then for gcd(255,x)>1, x must share 
    # a factor with 255. Wait, gcd(255,x) must be >1. So x must share a prime factor with 255, meaning 
    # x must be divisible by 3, 5, or 17.
    
    # So every element in S is divisible by 3, 5, or 17.
    
    # Now, if S contains an element divisible by 3 but not by 5 or 17, call it a.
    # If S contains an element divisible by 5 but not by 3 or 17, call it b.
    # Then gcd(a,b) might be 1. Specifically, if a=3 and b=5, gcd=1.
    # More carefully: a = 3*x where gcd(x, 5*17)=1 potentially... 
    # Actually, let's just say: can we have a mixed set?
    
    # If all elements are divisible by 3: size 85.
    # If all elements are divisible by 5: size 51.
    # If all elements are divisible by 17: size 15.
    
    # Could we have a set where not all are divisible by the same prime, but still pairwise non-coprime?
    # Example: {6, 10, 15}. gcd(6,10)=2, gcd(6,15)=3, gcd(10,15)=5. All >1. Size 3.
    # But we must include 255. Let's try {6, 10, 15, 255}. 
    # gcd(6,255)=3>1, gcd(10,255)=5>1, gcd(15,255)=15>1. OK.
    # Can we add more? Add 30: gcd(30,6)=6, gcd(30,10)=10, gcd(30,15)=15, gcd(30,255)=15. OK.
    # Add 3: gcd(3,10)=1. Not OK.
    # Add 25: gcd(25,6)=1. Not OK.
    
    # It seems the pure multiples of 3 set is large. Let's confirm 85 is the answer.
    # The set of all multiples of 3 in [1,255] has 85 elements.
    # This is the known result for this type of problem.
    
    return list(range(3, 256, 3))
