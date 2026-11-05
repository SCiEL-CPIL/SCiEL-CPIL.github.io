import math
import time
from itertools import combinations

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs.
    
    Strategy:
    1. Start with the set of all multiples of 3 (size 85).
    2. Identify candidates from [1, 255] not in the base set but sharing factors with 255 (3, 5, 17).
    3. Use a greedy/branch-and-bound heuristic to add candidates to the base set if they are compatible 
       (gcd > 1) with all existing members.
    4. This explores the "family" of the base solution by adding elements that maintain the pairwise property.
    """
    
    n = 255
    deadline = time.time() + 45  # Safety margin for 90s limit
    
    # Base set: multiples of 3
    base_set = list(range(3, n + 1, 3))
    base_set_set = set(base_set)
    
    # Candidates: numbers in [1, 255] not in base_set that share a factor with 255
    # 255 = 3 * 5 * 17
    # A number x is a candidate if x is not divisible by 3, but is divisible by 5 or 17.
    # Also, x must be compatible with 255 (which it is, since 255 is divisible by 5 and 17).
    candidates = []
    for x in range(1, n + 1):
        if x in base_set_set:
            continue
        if x % 5 == 0 or x % 17 == 0:
            candidates.append(x)
    
    # We want to add as many candidates as possible to base_set such that
    # all pairs in the new set have gcd > 1.
    # Note: Any two multiples of 3 already have gcd >= 3.
    # So we only need to check gcd(candidate, base_element) > 1 for each candidate added.
    # And gcd(candidate_i, candidate_j) > 1 for any two added candidates.
    
    # Greedy approach:
    # Sort candidates by "compatibility" with the current set (or just try to add them).
    # Since the base set is large, a candidate x is compatible with the base set
    # if gcd(x, b) > 1 for all b in base_set.
    # Since base_set contains all multiples of 3, let's analyze:
    # If x is a multiple of 5 but not 3:
    #   gcd(x, 3k) > 1 iff 5 divides 3k (impossible as 5!=3) OR x shares another factor with 3k.
    #   Actually, simpler: just compute.
    
    # Let's filter candidates that are compatible with the ENTIRE base set first.
    # If a candidate is not compatible with at least one element in the base set,
    # we cannot include it (unless we remove that base element, which is a different strategy).
    # The prompt says "KEEP this design", which implies starting from the multiples of 3.
    # Can we add a non-multiple of 3 to the set of multiples of 3?
    # Let S = {multiples of 3} U {x}.
    # For all m in S, gcd(m, x) > 1.
    # This requires gcd(x, 3k) > 1 for all k=1..85.
    # Let x = 5 * a. gcd(5a, 3k) > 1.
    # If a is not divisible by 3, then gcd(5a, 3) = gcd(5a, 3). If 3 does not divide 5a, then gcd(5a, 3) = 1.
    # So if x is not divisible by 3, we must have gcd(x, 3) > 1, which is impossible since 3 is prime and x not div by 3.
    # Wait. The base set contains 3.
    # gcd(x, 3) must be > 1.
    # This implies 3 divides x.
    # But we defined candidates as NOT in base_set (i.e., not divisible by 3).
    # Therefore, NO candidate outside the multiples of 3 can be added to the full set of multiples of 3,
    # because they would fail the gcd check with the element 3.
    
    # Conclusion: The set of all multiples of 3 is a "maximal" clique in the sense that
    # you cannot add any other number from [1, 255] to it without removing 3.
    # Could removing 3 allow adding more numbers?
    # The directive says "KEEP this design... investigate improvements within its family".
    # The "family" is the multiples of 3.
    # However, the inspiration 3 mentions "maximum clique search".
    # Let's verify if a different set of the same size or larger exists.
    # The size is 85.
    # Total numbers = 255.
    # Multiples of 3: 85.
    # Multiples of 5: 51.
    # Multiples of 17: 15.
    
    # If we stick to the "multiples of 3" family, the answer is 85.
    # Is it possible to get > 85?
    # Consider the set of all numbers divisible by 3 OR 5?
    # No, gcd(3, 5) = 1. So we can't have both 3 and 5.
    # We must pick a "dominant" factor or a structure where all pairs share a factor.
    
    # Let's try a different heuristic within the "search" family mentioned in inspiration 3,
    # but constrained to be efficient.
    # Actually, let's look at the "greedy clique construction" finding.
    # "Greedy clique construction starting from 255 may find a set larger than the 85 multiples of 3..."
    # This suggests that maybe we shouldn't keep ALL multiples of 3.
    # We can drop some multiples of 3 to include more non-multiples of 3, as long as pairwise gcd > 1 holds.
    
    # Let's implement a greedy set expansion starting from 255.
    # Start with S = {255}.
    # Iterate through candidates [1..255] \ S.
    # If candidate x is compatible with all y in S (gcd(x,y)>1), add it.
    # To maximize size, we should prioritize candidates that are "central" or have many connections.
    # A good heuristic for maximum clique is to order vertices by degree (descending).
    
    # Let's try this greedy approach. It is within the "search" family.
    
    # 1. Build compatibility graph
    # 2. Order candidates by degree in the graph (or just try adding in a smart order)
    
    # Given the small N=255, we can compute the full graph.
    
    # However, the deadline is tight. Let's do a fast greedy.
    
    # Candidates: All numbers 1..255.
    # We must include 255.
    
    # Let's use a greedy algorithm that adds nodes with the highest "potential" (degree in the current induced subgraph of compatible nodes).
    
    # Optimization: 
    # A number x is compatible with y if gcd(x,y) > 1.
    
    # Let's try a simple greedy:
    # 1. Start with S = {255}.
    # 2. Create a list of all other numbers 1..254.
    # 3. Sort them by descending number of prime factors (or just descending value? No, degree is better).
    # 4. For each candidate in sorted order:
    #    - Check if gcd(candidate, s) > 1 for all s in S.
    #    - If yes, add to S.
    
    # To speed up the check, we can maintain the set of prime factors of all elements in S.
    # Actually, checking gcd against all elements in S is O(|S| log N). |S| can be up to ~100. 255*100*20 ops is trivial.
    
    # Let's try this.
    
    # Sort candidates by "popularity" (number of divisors in [1,255] that share a factor with them, excluding themselves? Or just total divisors?)
    # Numbers with more prime factors are more likely to be compatible.
    
    def num_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    # Pre-compute compatibility?
    # Let's just use the greedy.
    
    # To improve on simple greedy, we can try multiple orderings or a simple backtracking if time permits.
    # But simple greedy is fast.
    
    # Let's try ordering by descending number of prime factors (with multiplicity).
    
    candidates_list = list(range(1, 256))
    candidates_list.remove(255)
    
    # Sort by descending number of prime factors
    candidates_list.sort(key=lambda x: -num_prime_factors(x))
    
    S = {255}
    
    for x in candidates_list:
        if time.time() > deadline:
            break
        # Check compatibility
        compatible = True
        for y in S:
            if math.gcd(x, y) == 1:
                compatible = False
                break
        if compatible:
            S.add(x)
    
    # The result might be smaller or larger than 85.
    # Let's compare with the base set of multiples of 3.
    
    base_size = 85
    greedy_size = len(S)
    
    if greedy_size > base_size:
        return sorted(S)
    else:
        return list(range(3, 256, 3))
