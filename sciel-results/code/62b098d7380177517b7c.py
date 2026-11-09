import math
import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    n = 210
    max_val = 10210
    min_val = 211
    
    # Target: Product * 210 is a perfect square.
    # Let P = prod(selected). P * 210 = k^2.
    # Let 210 = sf(210) * s1^2. P = sf(P) * s2^2.
    # sf(210) * sf(P) * (s1*s2)^2 = k^2.
    # Thus sf(210) * sf(P) must be a perfect square.
    # This implies sf(P) must be equal to sf(210).
    
    def get_squarefree_kernel(x):
        """
        Returns the square-free part of x.
        x = k^2 * sf(x)
        """
        sf = 1
        temp = x
        # Check factor 2
        if temp % 2 == 0:
            cnt = 0
            while temp % 2 == 0:
                temp //= 2
                cnt += 1
            if cnt % 2 == 1:
                sf *= 2
        # Check odd factors
        f = 3
        while f * f <= temp:
            if temp % f == 0:
                cnt = 0
                while temp % f == 0:
                    temp //= f
                    cnt += 1
                if cnt % 2 == 1:
                    sf *= f
            f += 2
        if temp > 1:
            sf *= temp
        return sf

    target_kernel = get_squarefree_kernel(n)
    
    # We need to select distinct integers from [211, 10210]
    # such that the product of their square-free kernels is a perfect square.
    # Wait, the condition is sf(P) = sf(210).
    # Let selected be {x1, ..., xk}.
    # P = x1 * ... * xk.
    # sf(P) = sf(x1 * ... * xk).
    # Note that sf(a*b) is not simply sf(a)*sf(b).
    # However, if we represent each number by its prime signature (exponents mod 2),
    # the condition is that the XOR (sum mod 2) of the signatures of the selected numbers
    # equals the XOR of the signature of 210.
    # Actually, simpler:
    # P * 210 is a square.
    # Let v_p(x) be the exponent of prime p in x.
    # v_p(P) + v_p(210) must be even for all p.
    # v_p(P) = sum_{x in selected} v_p(x).
    # So sum_{x in selected} v_p(x) ≡ v_p(210) (mod 2).
    
    # This is a subset sum problem over GF(2) for each prime.
    # Since we want to minimize max(selected), we should look at small numbers first.
    # A greedy approach:
    # 1. Precompute the "signature" (vector of exponents mod 2) for each number in range.
    #    Actually, we just need to track the "current parity vector" of the product.
    #    Let current_sig be the XOR of signatures of selected numbers.
    #    We want current_sig XOR target_sig == 0, i.e., current_sig == target_sig.
    
    # To make this efficient, we can map the signature to a hashable object.
    # The primes involved are up to 10210. There are ~1200 primes.
    # A dictionary mapping prime -> exponent mod 2 is feasible.
    # Or, since we only care about equality, we can use the square-free kernel itself?
    # No, sf(a)*sf(b) is not the kernel of a*b.
    # Example: a=2, b=8. sf(2)=2, sf(8)=2. sf(16)=1. 2*2=4 != 1.
    # So we must track prime parities.
    
    # However, there is a simpler observation.
    # If we pick numbers such that their product is a square, that's easier.
    # But we need P * 210 to be a square.
    # This is equivalent to P being "square-equivalent" to 210.
    
    # Let's use a state-space search or greedy matching.
    # Given the small size (100 items max), and the range (10k items),
    # we can try to find a small subset (e.g., size 2, 3, 4) that satisfies the condition.
    # If we find a subset of size k with max value M, that's a candidate.
    # We want to minimize M.
    
    # Strategy:
    # Iterate possible max values M from 211 upwards.
    # For each M, consider the set of available numbers S_M = {x | 211 <= x <= M}.
    # Check if there exists a subset of S_M (size <= 100) that satisfies the condition.
    # Since we want distinct integers, and we can pick up to 100,
    # but typically a small subset (2-4) is enough to fix the parities.
    
    # Why? The vector space of parities is high dimensional, but the numbers are "dense".
    # Actually, finding a subset of size 2 is like finding x, y in S_M such that
    # sig(x) XOR sig(y) == target_sig.
    # This is a collision search in the space of signatures.
    
    # Let's precompute signatures for all numbers up to 10210.
    # Signature representation: frozenset of primes with odd exponent.
    
    def get_prime_parity_set(x):
        """
        Returns a frozenset of primes that have an odd exponent in x.
        """
        ps = set()
        temp = x
        if temp % 2 == 0:
            cnt = 0
            while temp % 2 == 0:
                temp //= 2
                cnt += 1
            if cnt % 2 == 1:
                ps.add(2)
        f = 3
        while f * f <= temp:
            if temp % f == 0:
                cnt = 0
                while temp % f == 0:
                    temp //= f
                    cnt += 1
                if cnt % 2 == 1:
                    ps.add(f)
            f += 2
        if temp > 1:
            ps.add(temp)
        return frozenset(ps)

    # Precompute for all numbers in range
    # This might be slow if done naively for 10k numbers, but 10k is small.
    # 10k * sqrt(10k) ~ 1e6 ops. Fast enough.
    
    parities = {}
    for i in range(min_val, max_val + 1):
        parities[i] = get_prime_parity_set(i)
        
    target_parity = get_prime_parity_set(n)
    
    # We want to find a subset {x1, ..., xk} such that
    # XOR(parity[x1], ..., parity[xk]) == target_parity.
    
    # Greedy approach:
    # Try to find a pair (x, y) such that parity[x] XOR parity[y] == target_parity.
    # This is equivalent to parity[y] == parity[x] XOR target_parity.
    
    # We iterate M from min_val to max_val.
    # Maintain a set of parities seen so far?
    # No, we need to ensure distinct integers.
    # If we just check existence, we can do:
    # For a fixed M, let S = {x | min_val <= x <= M}.
    # We want to know if there exist distinct x, y in S such that parity[x] ^ parity[y] == target.
    
    # We can do this by scanning M from min_val upwards.
    # Keep a dictionary: parity -> list of numbers with that parity (or just one, to check distinctness later).
    # Actually, if we find a parity P that matches required, we need two distinct numbers.
    
    # Let's refine:
    # Iterate M from min_val to max_val.
    # Add M to our "pool".
    # Check if we can form the target parity using numbers from the pool.
    # Start with size 1:
    #   Does any x in pool have parity[x] == target?
    #   If yes, return [x] (wait, size 1 is allowed? "1..100 distinct integers". Yes.)
    #   But wait, if size 1, x * 210 is square. So x must be 210 * k^2.
    #   Smallest such x > 210 is 210 * 4 = 840.
    #   Let's check if 840 is in range. Yes.
    #   So [840] is a valid solution with max 840.
    #   Can we do better?
    #   If we use size 2: x, y. x*y*210 is square.
    #   This allows smaller max values potentially?
    #   Example: x=211, y=?
    #   parity[211] = {211}.
    #   We need parity[y] such that {211} ^ parity[y] == target_parity.
    #   target_parity for 210 = 2*3*5*7. Parity set {2,3,5,7}.
    #   So parity[y] must be {211, 2, 3, 5, 7}.
    #   y must be of the form 2^a * 3^b * 5^c * 7^d * 211^e * (square).
    #   Smallest y with this parity: 2*3*5*7*211 = 210 * 211 = 44310.
    #   This is > 10210. So no pair with 211 works within range.
    
    #   What if we pick x, y closer to 210?
    #   Let's just implement the search for minimal M.
    
    # Data structure:
    # parity_to_numbers: dict mapping frozenset -> list of numbers with that parity.
    # We add numbers one by one as M increases.
    
    parity_map = {}
    
    for M in range(min_val, max_val + 1):
        if time.time() > deadline:
            # Fallback: return a known valid solution if we have one, or a safe guess
            # We haven't found one yet in this loop, so we should have found one quickly.
            # If we time out, it's a failure.
            # But 10k iterations is fast.
            pass
            
        p_M = parities[M]
        
        if p_M not in parity_map:
            parity_map[p_M] = []
        parity_map[p_M].append(M)
        
        # Check if we can form target_parity using numbers in parity_map.
        
        # Case 1: Single number
        if p_M == target_parity:
            return [M]
            
        # Also check if any previous number had target_parity
        # (This is covered by the above if we check all, but we only check M.
        #  We should check if target_parity exists in parity_map)
        if target_parity in parity_map:
            # We can just return the smallest number with that parity?
            # No, we want to minimize MAX.
            # If we found a number x with parity target, max is x.
            # Since we are iterating M upwards, the first time we see target_parity
            # is the smallest such number.
            # So if we hit this condition, we can return that number.
            # But wait, a pair might have a smaller max?
            # No, if we have a single number x, max is x.
            # If we have a pair x, y, max is max(x,y).
            # If max(x,y) < x, that's impossible since y < x and x is the single number candidate?
            # Not necessarily.
            # Example: Single number 840. Max 840.
            # Pair: 211 and 44310 (out of range).
            # Pair: 212 and ?
            # 212 = 4 * 53. Parity {53}.
            # Need y with parity {53, 2, 3, 5, 7}.
            # Smallest y: 2*3*5*7*53 = 210 * 53 = 11130 (out of range).
            
            # It seems single number 840 is a strong candidate.
            # Are there pairs with max < 840?
            # We need x, y < 840 such that parity[x] ^ parity[y] == target.
            
            # Let's just let the loop run. It will find the first M where a solution exists.
            # But we need to check ALL combinations in the pool, not just involving M.
            # However, if a solution existed in the pool before M, we would have returned earlier?
            # No, we only check if M is part of a solution or if a single number works.
            # We need to check if ANY pair in the pool works.
            
            # This is getting complicated to do incrementally for pairs.
            # Given the small search space (10k), we can just precompute all pairs?
            # 10k^2 = 100M pairs. Too slow in Python.
            
            # Alternative:
            # For each M, check if there exists x < M such that parity[x] ^ parity[M] == target.
            # This is a lookup in parity_map.
            # required_parity = parity[M] ^ target_parity.
            # If required_parity in parity_map:
            #     Get a number x from parity_map[required_parity].
            #     If x != M (distinctness):
            #         return [x, M] (max is M)
            
            # This checks all pairs (x, M) where x < M.
            # This covers all pairs!
            
            req_p = None
            # XOR of sets: symmetric difference
            req_p = p_M ^ target_parity
            
            if req_p in parity_map:
                # Find a number x in parity_map[req_p]
                # We need x != M.
                # Since we added M to parity_map[p_M], and req_p might be p_M?
                # If req_p == p_M, then target_parity is empty set.
                # target_parity is not empty (210 is not a square).
                # So req_p != p_M.
                # Thus x comes from a different parity bucket, so x != M is guaranteed?
                # Wait, could x be M?
                # M is in bucket p_M. x is in bucket req_p.
                # If p_M != req_p, then M is not in bucket req_p.
                # So x != M is guaranteed.
                
                x = parity_map[req_p][0] # Any number works, we just need existence
                # We want to minimize max. Since we iterate M upwards,
                # the first M for which a valid x exists gives the minimal max.
                # Is this true?
                # Suppose there is a pair (a, b) with max(a,b) = M*.
                # When loop reaches M*, it checks if there is x < M* such that x^M* = target.
                # If a < b = M*, then x=a is in the map. So it will find it.
                # If b < a = M*, then x=b is in the map.
                # So yes, the first M found is the minimal max for pairs.
                
                # But we also need to consider single numbers.
                # If a single number x works, max is x.
                # If a pair (x, y) works with max < x, we would have found the pair earlier.
                # So we just need to check:
                # 1. Is there a single number in pool with parity target?
                # 2. Is there a pair in pool with max <= current M?
                
                # The logic above:
                # - Checks single number: if p_M == target, return [M].
                #   But what if a previous number had target parity?
                #   We should check `if target_parity in parity_map: return [min(parity_map[target_parity])]`
                #   Actually, if we found a single number earlier, we would have returned.
                #   So if we are still looping, no single number < M worked.
                #   So if p_M == target, M is the smallest single number.
                
                # - Checks pair ending at M.
                #   If found, max is M.
                #   Is it possible that a pair (x, y) with max < M exists but we missed it?
                #   No, because we would have returned when the loop reached max(x,y).
                
                # So, if we find a pair here, we return [x, M].
                return [x, M]
                
    # If no solution found in range (should not happen), fallback
    return [4 * n]

# Test helper
if __name__ == "__main__":
    res = construct()
    prod = 1
    for r in res:
        prod *= r
    total = prod * 210
    root = math.isqrt(total)
    print(res)
    print(root * root == total)
