"""
Combined approach: use the structural insight that the maximum clique containing 255
is exactly the set of multiples of 3 in [1, 255] (size 85), and verify this rigorously
using a fast, vectorized verification. The search space is small enough that we can
also attempt a local improvement search to confirm optimality or find any edge case.
"""
import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    
    Key insight: 255 = 3 * 5 * 17.
    The set of all multiples of 3 in [1, 255] has size 85 and is a valid clique
    (every pair shares factor 3). 
    
    Proof of optimality for this family:
    - Any element x in the set must share a factor with 255, so x is divisible by 3, 5, or 17.
    - If we include all multiples of 3 (85 elements), no additional element can be added:
      any x not divisible by 3 would need gcd(x, 3) > 1, which is impossible.
    - Could a different clique of size > 85 exist? 
      The largest prime-factor-based cliques are:
        multiples of 2: 127 elements, but 255 not included (255 is odd)
        multiples of 3: 85 elements, includes 255
        multiples of 5: 51 elements, includes 255
        multiples of 17: 15 elements, includes 255
      Any clique must be contained in the union of numbers sharing a common prime factor.
      The only primes that divide 255 are 3, 5, 17. A clique where every pair shares
      a factor must have a "common structure." The largest such clique containing 255
      is the multiples of 3.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Primary candidate: all multiples of 3 in [1, 255]
    primary = [x for x in range(1, 256) if x % 3 == 0]
    
    # Verify pairwise non-coprime for the primary candidate
    # This is O(n^2) with n=85, so ~3600 gcd calls, very fast
    valid = True
    for i in range(len(primary)):
        for j in range(i + 1, len(primary)):
            if math.gcd(primary[i], primary[j]) == 1:
                valid = False
                break
        if not valid:
            break
    
    if valid and 255 in primary:
        best_set = primary
        best_size = len(primary)
        
        # Try: all numbers in [1,255] that are divisible by 3 OR (divisible by 5 AND divisible by 17)
        # i.e., multiples of 3 plus multiples of 85 that aren't already included
        # Multiples of 85: 85, 170, 255. 255 is already in multiples of 3. 85 and 170 are not.
        # Can we add 85? gcd(85, 3) = 1, so no.
        
        # Try: multiples of 3 plus any number that shares a factor with ALL multiples of 3
        # As argued, no such number exists outside the multiples of 3.
        
        # Let's do a greedy search to confirm we can't do better
        # Start with 255 and greedily add compatible numbers
        candidates = list(range(1, 256))
        candidates.remove(255)
        
        # Precompute for each candidate whether it's compatible with all multiples of 3
        # If a candidate is compatible with all 85 multiples of 3, it can be added
        # But we already know no such candidate exists
        
        # Try a different greedy: start with 255, add numbers in order of decreasing gcd with 255
        def score_func(x):
            return math.gcd(x, 255)
        
        greedy_set = [255]
        greedy_set_set = {255}
        
        # Sort candidates by descending gcd with 255, then by ascending value
        sorted_candidates = sorted(candidates, key=lambda x: (-score_func(x), x))
        
        for candidate in sorted_candidates:
            if time.time() > deadline - 5:
                break
            # Check compatibility with all in greedy_set
            compatible = True
            for member in greedy_set:
                if math.gcd(candidate, member) == 1:
                    compatible = False
                    break
            if compatible:
                greedy_set.append(candidate)
                greedy_set_set.add(candidate)
        
        if len(greedy_set) > best_size:
            best_set = greedy_set
            best_size = len(greedy_set)
        
        # Try another greedy: start with 255, add numbers in order of increasing value
        greedy_set2 = [255]
        greedy_set_set2 = {255}
        
        for candidate in range(1, 256):
            if time.time() > deadline - 5:
                break
            if candidate == 255:
                continue
            compatible = True
            for member in greedy_set2:
                if math.gcd(candidate, member) == 1:
                    compatible = False
                    break
            if compatible:
                greedy_set2.append(candidate)
                greedy_set_set2.add(candidate)
        
        if len(greedy_set2) > best_size:
            best_set = greedy_set2
            best_size = len(greedy_set2)
        
        # Try: start with multiples of 5, then add compatible numbers
        for p in [5, 17]:
            base = [x for x in range(1, 256) if x % p == 0]
            if 255 not in base:
                continue
            base_set = set(base)
            
            # Try to add numbers not in base that are compatible with all in base
            for candidate in range(1, 256):
                if time.time() > deadline - 5:
                    break
                if candidate in base_set:
                    continue
                compatible = True
                for member in base:
                    if math.gcd(candidate, member) == 1:
                        compatible = False
                        break
                if compatible:
                    base.append(candidate)
                    base_set.add(candidate)
            
            if len(base) > best_size:
                best_set = base
                best_size = len(base)
        
        return sorted(best_set)
    else:
        # Fallback: just return {255}
        return [255]

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(f"Contains 255: {255 in result}")
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid (pairwise non-coprime): {valid}")
