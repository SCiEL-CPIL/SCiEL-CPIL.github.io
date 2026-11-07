"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def construct():
    """
    Find 1..100 distinct integers strictly larger than 210 and at most 10210.
    Their product multiplied by 210 must be a perfect square.
    Minimize max(selected integers) - 210.
    """
    start_time = time.time()
    deadline = start_time + 85.0

    n = 210
    max_val = 10210
    max_size = 100

    # 210 = 2 * 3 * 5 * 7
    # We need product S to have square-free part exactly 210
    # i.e., parity vector of product must match parity vector of 210

    def is_valid_solution(nums):
        """Check if 210 * prod(nums) is a perfect square."""
        prod = 210
        for x in nums:
            prod *= x
        # Check perfect square using integer sqrt
        s = math.isqrt(prod)
        return s * s == prod

    # Parity set of 210 = {2, 3, 5, 7}
    target = frozenset({2, 3, 5, 7})

    def parity_set(x):
        """Return frozenset of primes with odd exponent in x."""
        primes = set()
        temp = x
        d = 2
        while d * d <= temp:
            cnt = 0
            while temp % d == 0:
                cnt += 1
                temp //= d
            if cnt % 2 == 1:
                primes.add(d)
            d += 1 if d == 2 else 2
        if temp > 1:
            primes.add(temp)
        return frozenset(primes)

    # Known: [840] works, giving max=840, raw=630
    best_solution = [840]
    best_max = 840

    # Strategy: Search for solutions with smaller max
    # For each upper bound U, check if there exists a valid set with all elements in (210, U]
    
    # Precompute parity sets up to a reasonable limit
    # We'll search up to best_max - 1 initially, then expand if needed
    
    # Phase 1: Search for pairs with max < best_max
    # For pairs: need a, b in (210, U] with parity_set(a) ^ parity_set(b) = target
    # i.e., parity_set(b) = parity_set(a) ^ target
    
    # Build map from parity set to list of numbers
    ps_to_nums = defaultdict(list)
    parity_map = {}
    
    # Search up to best_max - 1
    search_limit = min(best_max - 1, max_val)
    
    for x in range(211, search_limit + 1):
        if time.time() > deadline - 10:
            break
        ps = parity_set(x)
        parity_map[x] = ps
        ps_to_nums[ps].append(x)
    
    # Sort numbers in each group
    for ps in ps_to_nums:
        ps_to_nums[ps].sort()
    
    # For each a, compute needed_b_ps = parity_set(a) ^ target
    # Then look for b > a with that parity set
    best_pair_max = float('inf')
    best_pair = None
    
    for a in range(211, search_limit + 1):
        if a not in parity_map:
            continue
        if time.time() > deadline - 10:
            break
        pa = parity_map[a]
        needed = pa ^ target
        if needed in ps_to_nums:
            # Find smallest b > a in ps_to_nums[needed]
            nums = ps_to_nums[needed]
            # Binary search for first element > a
            lo, hi = 0, len(nums)
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] <= a:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < len(nums):
                b = nums[lo]
                if b > a and b <= search_limit:
                    cur_max = max(a, b)
                    if cur_max < best_pair_max:
                        best_pair_max = cur_max
                        best_pair = (a, b)
                        # Check validity
                        if is_valid_solution([a, b]):
                            if cur_max < best_max:
                                best_max = cur_max
                                best_solution = [a, b]
    
    # Phase 2: If no pair found or we have time, search for triples
    # For triples: need a, b, c with pa ^ pb ^ pc = target
    # i.e., pc = pa ^ pb ^ target
    
    if time.time() < deadline - 15:
        triple_limit = min(best_max - 1, max_val)
        if triple_limit > 212:
            # Build map for numbers up to triple_limit
            ps_to_nums_t = defaultdict(list)
            parity_map_t = {}
            for x in range(211, triple_limit + 1):
                if time.time() > deadline - 15:
                    break
                ps = parity_set(x)
                parity_map_t[x] = ps
                ps_to_nums_t[ps].append(x)
            
            # Sort
            for ps in ps_to_nums_t:
                ps_to_nums_t[ps].sort()
            
            # For each pair (a, b), compute needed = pa ^ pb ^ target
            # Look for c > b with that parity
            best_triple_max = float('inf')
            best_triple = None
            
            # To limit work, only try pairs where max(a,b) < best_max
            # Also, we can prune: if a is already >= best_max, skip
            for a in range(211, min(triple_limit, best_max - 1) + 1):
                if a not in parity_map_t:
                    continue
                if time.time() > deadline - 15:
                    break
                pa = parity_map_t[a]
                
                # For b, we need b > a and b < best_max
                for b in range(a + 1, min(triple_limit, best_max - 1) + 1):
                    if b not in parity_map_t:
                        continue
                    if time.time() > deadline - 15:
                        break
                    pb = parity_map_t[b]
                    needed = pa ^ pb ^ target
                    if needed in ps_to_nums_t:
                        nums = ps_to_nums_t[needed]
                        # Find c > b
                        lo, hi = 0, len(nums)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if nums[mid] <= b:
                                lo = mid + 1
                            else:
                                hi = mid
                        if lo < len(nums):
                            c = nums[lo]
                            if c > b and c <= triple_limit:
                                cur_max = max(a, b, c)
                                if cur_max < best_triple_max:
                                    best_triple_max = cur_max
                                    best_triple = (a, b, c)
                                    if is_valid_solution([a, b, c]):
                                        if cur_max < best_max:
                                            best_max = cur_max
                                            best_solution = [a, b, c]
                                        break  # Found a valid triple for this a, move to next a
                    if time.time() > deadline - 15:
                        break
    
    # Phase 3: If still have time, try to find even better solutions
    # Check if we can find solutions with max < current best_max
    # This is already covered by the above phases
    
    # Final validation
    if is_valid_solution(best_solution):
        # Ensure all constraints met
        assert all(210 < x <= max_val for x in best_solution)
        assert len(set(best_solution)) == len(best_solution)
        assert 1 <= len(best_solution) <= max_size
        return [int(x) for x in best_solution]
    else:
        # Fallback to known good solution
        return [840]

# EVOLVE-BLOCK-END
