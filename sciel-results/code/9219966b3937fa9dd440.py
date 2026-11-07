import time
import math
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s
    n = 210
    max_val = 10210
    
    def is_square(m):
        if m < 0:
            return False
        s = math.isqrt(m)
        return s * s == m

    # Precompute square-free kernels for all numbers in range [211, 10210]
    # sf(x) is the product of primes p where the exponent of p in x is odd.
    # We can compute this efficiently by iterating through primes.
    
    # Generate primes up to sqrt(10210) ~ 101
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, prime in enumerate(sieve) if prime]
    
    primes = get_primes(101)
    
    # Map kernel -> list of numbers in [211, 10210] with that kernel
    # We only care about numbers > 210
    kernel_map = defaultdict(list)
    
    for x in range(211, max_val + 1):
        temp = x
        kernel = 1
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                kernel *= p
        if temp > 1:
            kernel *= temp
            
        kernel_map[kernel].append(x)
        
    # Sort lists to facilitate finding small max values
    for k in kernel_map:
        kernel_map[k].sort()
        
    # The condition is: 210 * a * b * c = square
    # Let sf(x) be the square-free part of x.
    # sf(210) = 210 (since 210 = 2*3*5*7 is square-free)
    # We need sf(210) * sf(a) * sf(b) * sf(c) to be a perfect square.
    # Let K = sf(210) = 210.
    # Let Ka, Kb, Kc be the kernels of a, b, c.
    # K * Ka * Kb * Kc = square.
    # This implies Ka * Kb * Kc = K * square.
    # Since K is square-free, Ka * Kb * Kc must be of the form K * s^2.
    # More simply, let's look at the parity of exponents.
    # For each prime p, let e_x be the exponent of p in x.
    # We need sum(e_x for x in {210, a, b, c}) to be even for all p.
    # This is equivalent to: sf(210) * sf(a) * sf(b) * sf(c) is a square.
    
    # Strategy:
    # Iterate through possible values of max_val_candidate (M) from 211 upwards.
    # We want to find a, b, c < M such that condition holds.
    # Actually, it's better to iterate through kernels.
    
    # Let's reframe: We want to minimize max(a,b,c).
    # Let M be the maximum. We check if there exist a, b in [211, M] distinct from each other and M
    # (or a,b,c all <= M) such that 210*a*b*c is a square.
    
    # Efficient search:
    # 1. Precompute all possible pairs of kernels (Ka, Kb) such that Ka * Kb is "compatible" with some Kc.
    # 2. For a fixed M, we can check if there is a triple.
    
    # Alternative: Since the range is small (10000), we can just iterate M from 211 to 10210.
    # For each M, we check if there exist a, b < M (or <= M, distinct) such that condition holds.
    # To make this fast, we can precompute a set of "required kernels".
    
    # Let's use the property: sf(a) * sf(b) * sf(c) = 210 * s^2.
    # Let target_kernel = 210.
    # We need sf(a) * sf(b) * sf(c) to be in the set of kernels { 210 * k^2 }.
    # Note that sf(x) is always square-free.
    # So sf(a) * sf(b) * sf(c) must be square-free? No, the product of square-free numbers is not necessarily square-free.
    # However, the condition is that the product is a multiple of 210 by a square.
    
    # Let's simplify:
    # For any triple (a, b, c), let P = sf(a) * sf(b) * sf(c).
    # We need P * 210 to be a perfect square.
    # This is equivalent to: P * 210 = Y^2.
    
    # Let's precompute for each pair of numbers (a, b) in our range, the "deficit" kernel D such that
    # if we pick c with sf(c) = D * k^2, then the product works.
    # Actually, simpler:
    # For a fixed pair (a, b), let R = sf(210) * sf(a) * sf(b).
    # We need sf(c) such that R * sf(c) is a square.
    # This means sf(c) must be equal to the square-free part of R.
    # Let req_kernel = sf(R).
    # Then we just need to check if there exists a number c in our range with sf(c) == req_kernel.
    
    # So the algorithm is:
    # 1. Precompute sf(x) for all x in [211, 10210].
    # 2. Group numbers by their sf(x).
    # 3. Iterate through all pairs (a, b) with a < b.
    # 4. Calculate req_kernel = sf(sf(210) * sf(a) * sf(b)).
    # 5. Check if there is a c in kernel_map[req_kernel] such that c is distinct from a and b.
    # 6. Track the minimum max(a, b, c).
    
    # To optimize:
    # We can iterate by the maximum element M.
    # Or just iterate all pairs and find the min max.
    # Number of pairs is ~ 10000^2 / 2 = 50 million. This might be too slow in Python if done naively.
    # We need to be smarter.
    
    # Let's iterate M from 211 to 10210.
    # Maintain a set of "available" numbers less than M.
    # For each new number M, check if it can form a triple with two existing numbers.
    # For M to be the max, we need a, b < M.
    # Condition: sf(M) * sf(a) * sf(b) * 210 = square.
    # Let Req = sf(210 * sf(M)).
    # We need sf(a) * sf(b) to have the same square-free part as Req.
    # i.e., sf(sf(a) * sf(b)) == Req.
    
    # Let's precompute a map: kernel -> list of numbers.
    # As we add M to the set of "small numbers", we can check if there's a pair in the small numbers that matches.
    
    # Actually, iterating M from 211 upwards:
    # Let S be the set of numbers in [211, M-1].
    # We want to know if there exist a, b in S such that sf(a) * sf(b) * sf(M) * 210 is a square.
    # Let K_M = sf(210 * sf(M)).
    # We need sf(a) * sf(b) to have square-free part K_M.
    # Let's maintain a dictionary `pair_kernels` which counts how many pairs in S have a specific square-free product.
    # Wait, the number of possible square-free products is large.
    
    # Better approach:
    # For each M, we check all a in S? No, that's O(N^2).
    
    # Let's try a different angle.
    # We found best_value = 630 with [840].
    # Can we do better with 3 numbers?
    # If we pick a, b, c close to 211, their product is ~ 211^3 ~ 9.4e6.
    # 210 * 9.4e6 ~ 2e9. sqrt is ~ 44000.
    # The constraints are loose.
    
    # Let's just do a randomized search or a structured search for small M.
    # Since we want to minimize max, we can iterate M from 211 to, say, 500.
    # For each M, we check all pairs a, b in [211, M-1].
    # If M is small (e.g., 300), the number of pairs is (300-211)^2 ~ 80^2 = 6400.
    # Total checks for M up to 500: sum_{M=211}^{500} (M-211)^2 ~ 300^3/3 ~ 9e6. This is feasible.
    
    best_value = 630
    best_result = [840]
    
    # Precompute sf for all numbers up to 10210 just in case, but we only need up to current M in the loop.
    # We already have kernel_map. Let's make a quick lookup for sf(x).
    sf_cache = {}
    for x in range(211, max_val + 1):
        temp = x
        kernel = 1
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count % 2 == 1:
                kernel *= p
        if temp > 1:
            kernel *= temp
        sf_cache[x] = kernel
        
    # We also need sf(210)
    sf_210 = 210 # 2*3*5*7
    
    # Iterate M
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
            
        # If M - 210 is not better than best_value, we can stop if we are sure no smaller max exists?
        # No, M is the max. If M - 210 >= best_value, we can break?
        # Yes, because we are iterating M in increasing order.
        if M - 210 >= best_value:
            break
            
        # Check if there exist a, b in [211, M-1] such that 210*a*b*M is a square.
        # Condition: sf(210) * sf(a) * sf(b) * sf(M) is a square.
        # Let K_target = sf(210 * sf(M)).
        # We need sf(a) * sf(b) to have square-free part K_target.
        
        K_M = sf_cache[M]
        # Calculate required square-free part for sf(a)*sf(b)
        # Let P = 210 * K_M. We need sf(P * sf(a) * sf(b)) = 1.
        # This is equivalent to sf(sf(a) * sf(b)) = sf(P).
        # Let Req = sf(P).
        
        # Compute Req
        temp_P = 210 * K_M
        req_kernel = 1
        for p in primes:
            if p * p > temp_P:
                break
            count = 0
            while temp_P % p == 0:
                count += 1
                temp_P //= p
            if count % 2 == 1:
                req_kernel *= p
        if temp_P > 1:
            req_kernel *= temp_P
            
        # Now we need to find a, b in [211, M-1] such that sf(a) * sf(b) has square-free part req_kernel.
        # This means sf(a) * sf(b) = req_kernel * k^2.
        
        # We can iterate a in [211, M-1].
        # For each a, we need sf(b) such that sf(a) * sf(b) = req_kernel * k^2.
        # Let sf(a) = Ka.
        # We need Ka * Kb = Req * k^2.
        # This implies Kb = Req * k^2 / Ka.
        # For Kb to be an integer, Ka must divide Req * k^2.
        # Also Kb must be a valid square-free kernel present in our set of numbers < M.
        
        # Simpler check:
        # For each a in [211, M-1]:
        #   Let Ka = sf(a).
        #   We need to check if there exists b in [211, M-1], b != a, such that sf(Ka * sf(b) * Req) == 1.
        #   This is equivalent to sf(Ka * sf(b)) == Req.
        #   Let X = Ka * Req.
        #   We need sf(X * sf(b)) == 1 => sf(b) == sf(X).
        #   So for each a, we compute X = Ka * Req, find req_b = sf(X), and check if there is a b < M with sf(b) == req_b.
        
        # To make this fast, we can precompute for the current M-1 range:
        # A set of kernels present in [211, M-1].
        # Actually, we can just maintain a list of numbers and their kernels as M grows.
        
        # Let's maintain `active_kernels` as a set of kernels found in [211, M-1].
        # But we also need to ensure distinctness. If req_b == Ka, we need at least two numbers with that kernel.
        # Let's maintain `kernel_counts` for the active range.
        
        # We'll rebuild or update incrementally.
        # Since M is small in the interesting range, we can just loop a.
        
        found = False
        # Optimization: If M is large, this loop is expensive.
        # But we break when M - 210 >= best_value.
        # Initial best_value is 630. So we check M up to 210 + 630 = 840.
        # Range size ~ 630. Pairs ~ 200k. Total operations ~ 630 * 200k = 126M. Might be tight in Python.
        
        # Let's optimize the inner loop.
        # Precompute a set of available kernels for the range [211, M-1].
        # Actually, we can just check existence.
        
        # Let's use the kernel_map we built earlier? No, that's for all numbers.
        # We need to know if a number with a specific kernel exists in [211, M-1].
        # We can precompute for each kernel, the smallest number > 210 with that kernel.
        # If min_num_for_kernel[k] < M, then such a number exists.
        
        # Let's build min_num_for_kernel
        min_num_for_kernel = {}
        for k, nums in kernel_map.items():
            if nums:
                min_num_for_kernel[k] = nums[0] # Since sorted, this is the smallest > 210
                
        # Now for each M:
        # For each a in [211, M-1]:
        #   Ka = sf(a)
        #   X = Ka * Req
        #   req_b = sf(X)
        #   Check if min_num_for_kernel.get(req_b, inf) < M
        #   If yes, we need to ensure we can pick b distinct from a.
        #   If req_b != Ka, then any b with kernel req_b is distinct from a (since a has kernel Ka).
        #   If req_b == Ka, we need to ensure there is another number with kernel Ka in [211, M-1].
        #   This requires knowing the second smallest number with kernel Ka.
        
        # Let's build second_min_num_for_kernel
        second_min_num_for_kernel = {}
        for k, nums in kernel_map.items():
            if len(nums) >= 2:
                second_min_num_for_kernel[k] = nums[1]
            else:
                second_min_num_for_kernel[k] = float('inf')
                
        # Helper to compute sf of a product quickly?
        # We have sf_cache for numbers, but X = Ka * Req might be large.
        # However, Ka and Req are square-free.
        # sf(Ka * Req) can be computed by combining their prime factors.
        # Since Ka and Req are small (product of primes < 101), we can just factorize them.
        
        def sf_of_product(k1, k2):
            # k1, k2 are square-free
            prod = k1 * k2
            # Factorize prod
            temp = prod
            res = 1
            for p in primes:
                if p * p > temp:
                    break
                count = 0
                while temp % p == 0:
                    count += 1
                    temp //= p
                if count % 2 == 1:
                    res *= p
            if temp > 1:
                res *= temp
            return res

        for a in range(211, M):
            if time.time() > deadline:
                break
            Ka = sf_cache[a]
            X = Ka * req_kernel
            req_b = sf_of_product(Ka, req_kernel) # Wait, sf(Ka * Req) is not necessarily sf_of_product if they share primes?
            # Yes, if they share primes, the exponents add. Since both are 0 or 1, sum is 0, 1, or 2.
            # If 2, it becomes 0 in square-free part.
            # My sf_of_product handles this correctly by factorizing the product.
            
            # Check if b exists
            min_b = min_num_for_kernel.get(req_b, float('inf'))
            if min_b < M:
                # Check distinctness
                if req_b != Ka:
                    # Any b with this kernel works, and it's distinct from a
                    found = True
                    break
                else:
                    # Need another number with same kernel Ka
                    second_min_b = second_min_num_for_kernel.get(Ka, float('inf'))
                    if second_min_b < M:
                        # Ensure the second min is not a itself?
                        # min_b is the smallest. If min_b == a, we need the second smallest.
                        # If min_b != a, we have at least two.
                        # Actually, if min_b < M and second_min_b < M, we have at least two candidates.
                        # We just need to pick one different from a.
                        # If a is the smallest, pick second. If a is not smallest, pick smallest.
                        found = True
                        break
                        
        if found:
            # We found a triple with max M.
            # We don't need to reconstruct the exact triple for the score, just the value.
            # But we need to return a valid list.
            # We should probably store the actual triple to return it.
            # Let's refine to store the best triple.
            
            # To avoid expensive reconstruction, let's just update best_value and store M.
            # At the end, we can re-run the check for the best M to get the actual numbers.
            # Or we can store the triple when we find it.
            
            # Let's store the triple.
            # We need to find a and b explicitly.
            # Re-do the loop for this M to find a, b.
            for a in range(211, M):
                Ka = sf_cache[a]
                X = Ka * req_kernel
                req_b = sf_of_product(Ka, req_kernel)
                min_b = min_num_for_kernel.get(req_b, float('inf'))
                if min_b < M:
                    if req_b != Ka:
                        b = min_b
                        c = M
                        # Verify
                        prod = 210 * a * b * c
                        if is_square(prod):
                            val = c - 210
                            if val < best_value:
                                best_value = val
                                best_result = [int(a), int(b), int(c)]
                        break
                    else:
                        second_min_b = second_min_num_for_kernel.get(Ka, float('inf'))
                        if second_min_b < M:
                            # Pick b such that b != a
                            if min_b == a:
                                b = second_min_b
                            else:
                                b = min_b
                            c = M
                            prod = 210 * a * b * c
                            if is_square(prod):
                                val = c - 210
                                if val < best_value:
                                    best_value = val
                                    best_result = [int(a), int(b), int(c)]
                            break
                            
            if best_value <= 5:
                break

    # Verify final result
    prod = 210
    for x in best_result:
        prod *= x
    if not is_square(prod):
        return [840]

    return best_result
