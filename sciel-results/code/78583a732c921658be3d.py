import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 85
    
    max_val = 10210
    min_val = 211
    
    # Primes up to 10210
    # We need to represent squarefree kernels as bitmasks.
    # The number of primes up to 10210 is approx 10210 / ln(10210) ~ 10210 / 9.2 ~ 1100.
    # Python integers can handle bitmasks of this size easily.
    
    # Sieve of Eratosthenes to get primes
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Map prime to index for bitmask
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def get_mask(n):
        mask = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                exp = 0
                while temp % d == 0:
                    temp //= d
                    exp += 1
                if exp % 2 == 1:
                    if d in prime_to_idx:
                        mask |= (1 << prime_to_idx[d])
            d += 1 if d == 2 else 2
        if temp > 1:
            if temp in prime_to_idx:
                mask |= (1 << prime_to_idx[temp])
        return mask

    # Precompute masks for all n in [min_val, max_val]
    masks = [0] * (max_val + 1)
    for n in range(min_val, max_val + 1):
        masks[n] = get_mask(n)
        
    target_mask = get_mask(210)
    
    # Build map from mask to sorted list of numbers
    mask_to_nums = defaultdict(list)
    for n in range(min_val, max_val + 1):
        mask_to_nums[masks[n]].append(n)
        
    # Ensure lists are sorted (they are built in increasing n order)
    
    best_max = max_val + 1
    best_set = None
    
    # Size 1
    if target_mask in mask_to_nums:
        nums = mask_to_nums[target_mask]
        if nums:
            n = nums[0]
            if n < best_max:
                best_max = n
                best_set = [n]
                
    # Size 2
    # Iterate over all masks present
    # For each mask m, check if m ^ target_mask is present
    # To avoid double counting, we can just check all pairs of masks
    # But we want min max(n1, n2)
    # For a specific pair of masks (m1, m2), the best pair is (min(S_m1), min(S_m2))
    
    seen_mask_pairs = set()
    for m in mask_to_nums.keys():
        comp_m = m ^ target_mask
        if comp_m in mask_to_nums:
            # Avoid checking same pair twice
            pair_key = (min(m, comp_m), max(m, comp_m))
            if pair_key in seen_mask_pairs:
                continue
            seen_mask_pairs.add(pair_key)
            
            nums1 = mask_to_nums[m]
            nums2 = mask_to_nums[comp_m]
            
            if not nums1 or not nums2:
                continue
                
            n1 = nums1[0]
            n2 = nums2[0]
            
            # They are distinct because masks are different (target_mask != 0)
            # Wait, if m == comp_m, then target_mask == 0, which is false.
            # So n1 != n2 is guaranteed if lists don't overlap, which they don't.
            
            m_val = max(n1, n2)
            if m_val < best_max:
                best_max = m_val
                best_set = [n1, n2]
                
    # Size 3
    # Iterate x, y. Find z.
    # Prune if max(x, y) >= best_max
    # We only need to search x, y < best_max
    
    search_limit = best_max
    
    # To make it faster, we can iterate x from min_val to search_limit
    # and y from x+1 to search_limit
    
    # Precompute a quick lookup for min z?
    # For a given mask, we want the smallest number.
    # But we need z != x, z != y.
    # So if min(S_mask) == x or y, we need the next one.
    
    for x in range(min_val, search_limit):
        if time.time() > deadline - 5:
            break
        mx = masks[x]
        for y in range(x + 1, search_limit):
            if time.time() > deadline - 5:
                break
            if max(x, y) >= best_max:
                continue # Prune, though y < search_limit = best_max, so this is redundant if search_limit is tight
            my = masks[y]
            mz = target_mask ^ mx ^ my
            
            if mz in mask_to_nums:
                nums_z = mask_to_nums[mz]
                # Find smallest z in nums_z such that z != x and z != y
                z = None
                for candidate in nums_z:
                    if candidate != x and candidate != y:
                        z = candidate
                        break
                
                if z is not None:
                    m_val = max(x, y, z)
                    if m_val < best_max:
                        best_max = m_val
                        best_set = [x, y, z]
                        # Update search limit to new best
                        search_limit = best_max
                        # Note: changing search_limit inside loop is tricky if we are iterating
                        # But since we break or continue based on current best, it's okay.
                        # Actually, if we find a better best, we can restart or just let it be.
                        # The inner loop will finish for this x.
                        # The outer loop condition uses old search_limit?
                        # It's better to just check `if max(x,y) >= best_max: break`
    
    if best_set is None:
        # Fallback, should not happen
        best_set = [840]
        best_max = 840
        
    # Verify
    prod = 210
    for x in best_set:
        prod *= x
        
    x_sqrt = math.isqrt(prod)
    if x_sqrt * x_sqrt != prod:
        # Fallback
        best_set = [840]
        
    return [int(x) for x in best_set]
