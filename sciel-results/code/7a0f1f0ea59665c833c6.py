import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s
    
    def is_square(m):
        if m < 0:
            return False
        s = math.isqrt(m)
        return s * s == m

    # 1. Precompute square-free kernels for all integers in (210, 10210]
    # The square-free kernel of n is the product of primes dividing n with odd exponent.
    # If n = a^2 * k, then k is the square-free kernel.
    # For n * 210 to be a square, sf(n) * sf(210) must be a square.
    # sf(210) = 2 * 3 * 5 * 7 = 210 (since 210 is square-free).
    # So we need sf(n) * 210 to be a square.
    # Let sf(n) = k. Then k * 210 = m^2.
    # This implies k = m^2 / 210. Since k is square-free, this restricts possible k values.
    # Alternatively, we can just store sf(n) and check compatibility.
    
    # A more robust way: 
    # Product P of selected numbers * 210 is a square.
    # Let P = n1 * n2 * ... * nk.
    # P * 210 = S^2.
    # This is equivalent to saying that the product of the square-free parts of (n1, ..., nk, 210) is 1 (in the group of square-free integers under multiplication mod squares, which is isomorphic to (Z/2Z)^r).
    # Actually, simpler: 
    # For a single number n: n * 210 = square => n = 210 * t^2.
    # For two numbers n1, n2: n1 * n2 * 210 = square.
    # Let sf(x) be the square-free part of x.
    # sf(n1 * n2 * 210) = 1.
    # Since sf is multiplicative up to squares: sf(n1)*sf(n2)*sf(210) must be a square.
    # sf(210) = 210.
    # So sf(n1)*sf(n2)*210 = square.
    # Let A = sf(n1), B = sf(n2). A*B*210 = square.
    # This means A*B = 210 * (square).
    # Since A and B are square-free, A*B is square-free only if gcd(A,B)=1.
    # If they share factors, those factors cancel out to even powers.
    # Actually, the condition "A*B*210 is a square" is equivalent to "A*B*210 has no prime with odd exponent".
    # This is equivalent to: for every prime p, exp_p(A) + exp_p(B) + exp_p(210) is even.
    # Since A, B, 210 are square-free, exp_p is 0 or 1.
    # This is a linear algebra problem over GF(2).
    # But we can just use the property: A*B*210 is a square iff A*B = 210 * k^2 for some integer k?
    # No. If A=2, B=105 (which is 3*5*7). A*B*210 = 2*105*210 = 2*105*2*105 = (2*105)^2. Square.
    # Here A*B = 210.
    # If A=1, B=210. A*B*210 = 210*210 = square. A*B = 210.
    # If A=2, B=2. A*B*210 = 4*210 = not square. A*B = 4.
    # So the condition is not simply A*B=210.
    # The condition is that the symmetric difference of the prime sets of A, B, and 210 is empty.
    # i.e., Set(A) XOR Set(B) XOR Set(210) = Empty Set.
    # This implies Set(A) XOR Set(B) = Set(210).
    
    # Let's represent each number by its square-free kernel as a tuple of primes or just the integer value.
    # Since we only care about the square-free part, and the square-free part is unique,
    # we can map each number n to sf(n).
    
    # Precompute sf(n) for n in 211..10210
    max_val = 10210
    sf = [0] * (max_val + 1)
    
    # Sieve-like approach to compute square-free parts
    # Initialize sf[i] = i
    for i in range(2, max_val + 1):
        sf[i] = i
        
    for p in range(2, max_val + 1):
        if sf[p] == p: # p is prime
            p2 = p * p
            for multiple in range(p2, max_val + 1, p2):
                sf[multiple] //= p2

    # Now sf[n] is the square-free kernel of n.
    
    # We want to find a set S of distinct integers in (210, 10210] such that
    # product(S) * 210 is a perfect square.
    # This is equivalent to: The XOR of the prime factorization vectors of sf(n) for n in S,
    # combined with the vector for 210, results in the zero vector.
    # Or simply: The product of sf(n) for n in S, multiplied by 210, is a perfect square.
    
    # Let's build a dictionary mapping sf(n) to the list of numbers n having that sf.
    # We want to minimize max(S) - 210.
    # So we should search for solutions with small max(S).
    
    # Strategy:
    # 1. Try sets of size 1: n * 210 = square => n = 210 * t^2.
    #    Check if n <= 10210.
    #    210 * 1^2 = 210 (not > 210)
    #    210 * 2^2 = 840. 840 <= 10210. So [840] is a valid solution. max=840. cost=840-210=630.
    #    210 * 3^2 = 1890. cost=1680.
    #    ...
    #    The best size-1 solution is 840.
    
    # 2. Try sets of size 2: n1 * n2 * 210 = square.
    #    Condition: sf(n1) * sf(n2) * 210 is a square.
    #    Let k1 = sf(n1), k2 = sf(n2).
    #    We need k1 * k2 * 210 to be a square.
    #    This implies k1 * k2 = 210 * (some square).
    #    Actually, let's just iterate through possible pairs of kernels.
    #    But there are many kernels.
    
    # Better approach for size 2:
    # For each n1 in (210, 10210], we need to find n2 such that sf(n1)*sf(n2)*210 is a square.
    # This is equivalent to sf(n2) = sf(n1) * 210 / (square).
    # Wait, sf(n1)*sf(n2)*210 = m^2.
    # Since sf(n1) and sf(n2) are square-free, let's analyze the prime exponents.
    # Let v_p(x) be exponent of p in x.
    # v_p(sf(n1)) + v_p(sf(n2)) + v_p(210) = 0 mod 2.
    # v_p(sf(n2)) = v_p(sf(n1)) + v_p(210) mod 2.
    # So the square-free kernel of n2 is determined by the square-free kernel of n1 and 210.
    # Specifically, if we represent square-free numbers as bitmasks of primes,
    # mask(n2) = mask(n1) XOR mask(210).
    # But we don't have a fixed prime list for bitmasking easily without knowing all primes.
    # However, we can just compute the required sf(n2) value.
    # Let A = sf(n1).
    # We need B = sf(n2) such that A * B * 210 is a square.
    # This means A * B = 210 * k^2 for some integer k?
    # No, A*B*210 = square => A*B = square / 210.
    # Since A and B are square-free, A*B is square-free only if gcd(A,B)=1.
    # If they share factors, those factors appear with even exponent in A*B? No, A and B are square-free, so if p|A and p|B, then p^2 | A*B.
    # So A*B is not necessarily square-free.
    # Let's just calculate the required B directly.
    # A * B * 210 = S^2.
    # B = S^2 / (210 * A).
    # For B to be an integer, 210*A must divide S^2.
    # This implies that for each prime p, v_p(210*A) must be even? No.
    # S^2 is a perfect square. So v_p(S^2) is even.
    # Therefore v_p(210*A) must be even for B to be an integer?
    # No, B is an integer, but B doesn't have to be square-free in the calculation, it just has to BE the square-free kernel of n2.
    # Actually, the condition "A * B * 210 is a square" where A and B are square-free integers
    # is equivalent to saying that the product A*B*210 has all even prime exponents.
    # Let's define a function `complement_kernel(A)` that returns the unique square-free integer B
    # such that A*B*210 is a square.
    # If no such B exists (which is impossible, we can always find one), or if B is not square-free?
    # Actually, for any square-free A, there is a unique square-free B such that A*B*210 is a square.
    # Proof: Let A = prod p_i. 210 = prod q_j.
    # We need B such that for each prime p, exp in A + exp in B + exp in 210 is even.
    # exp in B = exp in A + exp in 210 (mod 2).
    # Since exp in A and exp in 210 are 0 or 1, exp in B is 0 or 1.
    # So B is determined uniquely as the product of primes p where (v_p(A) + v_p(210)) is odd.
    # This B is square-free.
    
    def get_complement_kernel(A):
        # Returns the square-free kernel B such that A*B*210 is a square.
        # We need to factor A and 210.
        # But we can do this more efficiently.
        # Let's precompute the "mask" or just the value.
        # Actually, we can just compute B by iterating through primes?
        # Or simpler: 
        # A * B * 210 = square.
        # Let G = gcd(A, 210).
        # Let A' = A / G, 210' = 210 / G.
        # A' and 210' are coprime.
        # A * B * 210 = A' * G * B * 210' * G = A' * 210' * B * G^2.
        # For this to be a square, A' * 210' * B must be a square.
        # Since gcd(A', 210') = 1, A' and 210' must each be squares times a common factor?
        # No, A' and 210' are coprime. So A' must be a square and 210' must be a square?
        # Not necessarily. A' * 210' * B = square.
        # Since A' and 210' are coprime, let A' * 210' = C.
        # C * B = square.
        # This implies B = C * k^2? No, B is square-free.
        # This line of reasoning is getting messy. Let's just use the prime factorization definition.
        
        # Efficient way:
        # We want B such that sf(A * B * 210) = 1.
        # This is equivalent to sf(A) * sf(B) * sf(210) = 1 in the square-free group.
        # Which means sf(B) = sf(A) * sf(210) in that group.
        # The group operation is multiplication followed by removing square factors.
        # So B = sf(A * 210).
        # Let's verify: If B = sf(A * 210), then A * B * 210 = A * 210 * sf(A * 210).
        # Let X = A * 210. X = Y^2 * sf(X).
        # So A * 210 * sf(A * 210) = Y^2 * sf(X) * sf(X) = Y^2 * sf(X)^2 = (Y * sf(X))^2.
        # Yes! It is a perfect square.
        # So the required square-free kernel for n2 is simply sf(n1 * 210).
        
        return sf[A * 210] # Wait, A is sf(n1). A*210 might be large, but A <= 10210, so A*210 <= 2.1M.
        # We only precomputed sf up to 10210.
        # We need sf up to 10210 * 210 = 2144100.
        # That's a bit large for a precomputation array if we do it naively, but 2M is fine.
        # Or we can compute sf(A*210) on the fly using the precomputed sf for factors?
        # sf(x*y) is not simply sf(x)*sf(y) if they share factors.
        # But we can just extend the sieve to 2144100.
        pass

    # Let's extend the sieve to max_val * 210
    max_needed = max_val * 210
    # This might be slow if done from scratch.
    # Alternative: Just compute sf(A*210) using the property that sf(n) is the product of primes with odd exponent.
    # We can factor A*210 using the prime factors of A and 210.
    # But A is square-free.
    # Let's just precompute sf up to 2144100.
    
    # Re-run sieve for larger range
    sf_large = [0] * (max_needed + 1)
    for i in range(2, max_needed + 1):
        sf_large[i] = i
        
    # Primes up to sqrt(2144100) ~ 1464
    limit = int(math.isqrt(max_needed))
    for p in range(2, limit + 1):
        if sf_large[p] == p:
            p2 = p * p
            for multiple in range(p2, max_needed + 1, p2):
                sf_large[multiple] //= p2
                
    # Now sf_large[n] gives the square-free kernel for n up to 2144100.
    # And sf[n] gives it for n up to 10210.
    
    # Build map: kernel -> list of numbers in (210, 10210] with that kernel.
    kernel_map = defaultdict(list)
    for n in range(211, max_val + 1):
        k = sf[n]
        kernel_map[k].append(n)
        
    # We want to minimize max(S).
    # Let's iterate through possible max values M from 211 upwards.
    # For each M, we check if there is a valid set S with max(S) <= M.
    # Actually, we can just find the best solution for each size (1, 2, 3...) and take the min max(S).
    
    best_max = max_val + 1
    best_set = None
    
    # Size 1
    # n * 210 = square => n = 210 * t^2.
    # We already know 840 is a solution.
    # Let's check if there are any smaller?
    # t=1 -> 210 (not > 210)
    # t=2 -> 840.
    # So min max for size 1 is 840.
    if 840 <= max_val:
        if 840 < best_max:
            best_max = 840
            best_set = [840]
            
    # Size 2
    # We need n1, n2 such that sf(n1) * sf(n2) * 210 is a square.
    # This means sf(n2) = sf(n1 * 210).
    # Let k1 = sf(n1). Then k2 = sf_large[k1 * 210].
    # We need to find n1, n2 in (210, 10210] such that sf(n1)=k1, sf(n2)=k2.
    # To minimize max(n1, n2), we should look for small n1 and small n2.
    # We can iterate through all possible kernels k1 that appear in our range.
    # For each k1, compute k2.
    # If k2 exists in kernel_map, we have candidates.
    # We want to pick n1 from kernel_map[k1] and n2 from kernel_map[k2] such that n1 != n2 and max(n1, n2) is minimized.
    # Since we want to minimize the global max, we can just find the pair with the smallest max(n1, n2).
    
    # To do this efficiently:
    # For each kernel k1 in kernel_map:
    #   k2 = sf_large[k1 * 210]
    #   if k2 in kernel_map:
    #      if k1 == k2:
    #         We need two distinct numbers from kernel_map[k1].
    #         The best pair is the two smallest numbers in kernel_map[k1].
    #         max_pair = second_smallest.
    #      else:
    #         We need one from kernel_map[k1] and one from kernel_map[k2].
    #         The best pair is (smallest in k1, smallest in k2) or (smallest in k2, smallest in k1)?
    #         Actually, we want to minimize max(n1, n2).
    #         So we should pick the smallest n1 from list1 and smallest n2 from list2.
    #         Then max(n1, n2) is the candidate.
    #         But maybe a slightly larger n1 allows a much smaller n2?
    #         No, n2 is fixed by the kernel. The smallest n2 in the list is the best choice for n2.
    #         Similarly for n1.
    #         So the optimal pair for a given (k1, k2) is (min(list1), min(list2)).
    #         Wait, if k1 != k2, the sets are disjoint.
    #         So we just take min(list1) and min(list2).
    #         Candidate max is max(min(list1), min(list2)).
    
    # Let's precompute the minimum element for each kernel to speed this up.
    # Actually, we just need the two smallest for the k1==k2 case.
    
    min_for_kernel = {}
    second_min_for_kernel = {}
    for k, lst in kernel_map.items():
        if lst:
            min_for_kernel[k] = min(lst)
            if len(lst) > 1:
                # Find second min
                # Sorting a small list is fine, or just scan
                min1 = min(lst)
                min2 = float('inf')
                for x in lst:
                    if x < min2 and x != min1:
                        min2 = x
                second_min_for_kernel[k] = min2
            else:
                second_min_for_kernel[k] = float('inf')
                
    # Now iterate through kernels
    for k1 in kernel_map:
        k2 = sf_large[k1 * 210]
        if k2 in kernel_map:
            if k1 == k2:
                if k1 in second_min_for_kernel:
                    n1 = min_for_kernel[k1]
                    n2 = second_min_for_kernel[k1]
                    candidate_max = max(n1, n2)
                    if candidate_max < best_max:
                        best_max = candidate_max
                        best_set = [n1, n2]
            else:
                n1 = min_for_kernel[k1]
                n2 = min_for_kernel[k2]
                if n1 != n2: # They are from different kernels, so distinct
                    candidate_max = max(n1, n2)
                    if candidate_max < best_max:
                        best_max = candidate_max
                        best_set = [n1, n2]
                        
    # Size 3
    # n1 * n2 * n3 * 210 = square.
    # sf(n1)*sf(n2)*sf(n3)*210 = square.
    # Let k1, k2, k3 be the kernels.
    # k1*k2*k3*210 = square.
    # This implies k1*k2 = sf(k3 * 210) ? No.
    # k1*k2*k3 = sf(210) in the group?
    # k1*k2*k3*210 = square => k1*k2*k3 = sf(210 * square) = sf(210) = 210?
    # No. k1*k2*k3*210 = S^2 => k1*k2*k3 = S^2/210.
    # Since k1,k2,k3 are square-free, their product must be of the form 210 * (square).
    # Let's just iterate pairs (n1, n2) and check if there is a n3.
    # For a given n1, n2, we need n3 such that sf(n1*n2*n3*210)=1.
    # sf(n3) = sf(n1*n2*210).
    # Let K = sf(n1*n2*210).
    # We need to find n3 in kernel_map[K] such that n3 is distinct from n1, n2.
    # To minimize max(n1, n2, n3), we should iterate n1, n2 in increasing order of their max?
    # This is expensive.
    
    # Optimization:
    # We already found a size-2 solution. Let's see if size-3 can be better.
    # The best size-2 solution has max = best_max.
    # For size-3 to be better, max(n1, n2, n3) < best_max.
    # This implies n1, n2, n3 < best_max.
    # So we only need to consider numbers less than best_max.
    
    # Let's check if best_max is small enough to allow a size-3 search.
    # If best_max is, say, 300, we only check numbers in (210, 300).
    
    if best_max <= max_val:
        # Create a restricted kernel map for numbers < best_max
        restricted_kernel_map = defaultdict(list)
        for n in range(211, best_max): # n < best_max
            k = sf[n]
            restricted_kernel_map[k].append(n)
            
        # Iterate pairs (n1, n2) from the restricted set
        # To avoid O(N^2) if best_max is large, we hope best_max is small.
        # If best_max is large (e.g., 840), N ~ 630. Pairs ~ 200k.
        # For each pair, compute K = sf(n1*n2*210) and check if K is in restricted_kernel_map.
        # This is feasible.
        
        # Get list of numbers in restricted range
        restricted_nums = [n for n in range(211, best_max) if n in sf] # All are in sf
        # Actually, just range(211, best_max)
        
        # To be efficient, iterate n1, n2 such that n1 < n2 < best_max.
        # We want to find if there exists n3 < best_max, n3 != n1, n3 != n2, such that
        # sf(n1*n2*n3*210) = 1.
        # This is equivalent to sf(n3) = sf(n1*n2*210).
        
        # Precompute sf for products?
        # n1*n2*210 can be up to 10210^2 * 210 ~ 2e10. Too big for precomputed array.
        # We need to compute sf(n1*n2*210) on the fly.
        # We can factor n1, n2, 210.
        # Since n1, n2 are small (< 840), we can factor them quickly.
        
        def sf_on_the_fly(n):
            # Factor n and return square-free part
            # n is at most ~10000 * 10000 * 210 ~ 2e10
            # This might be slow if done for 200k pairs.
            # But we can optimize:
            # sf(n1*n2*210) = sf(n1) * sf(n2) * sf(210) with cancellation?
            # No, sf(a*b) is not sf(a)*sf(b).
            # However, we know sf(n1) and sf(n2) from our precomputed array.
            # Let A = sf(n1), B = sf(n2), C = 210.
            # We need sf(A*B*C * (squares from n1, n2)).
            # The square factors don't affect the kernel.
            # So sf(n1*n2*210) = sf(A*B*C).
            # A, B, C are square-free.
            # So we just need to compute the square-free part of the product of three square-free numbers.
            # This is just the product of primes that appear in an odd number of A, B, C.
            
            # We can do this by factoring A, B, C.
            # But A, B are small (<= 10210). C=210.
            # We can precompute prime factors for all numbers up to 10210?
            # Or just use the fact that A and B are square-free.
            
            # Let's just use the `sf_large` array if A*B*C fits.
            # A <= 10210, B <= 10210, C=210.
            # A*B*C <= 10210^2 * 210 ~ 2.1e10.
            # sf_large only goes to 2.1e6.
            # So we can't use sf_large directly.
            
            # We need a fast sf calculation for numbers up to 2e10.
            # Trial division up to sqrt(2e10) ~ 141421 is too slow for 200k calls.
            
            # Alternative:
            # We know A and B are square-free.
            # Let's represent A and B as sets of primes.
            # We can precompute the prime factors of all numbers up to 10210.
            
            pass
            
        # Precompute prime factors for numbers up to max_val
        prime_factors = [[] for _ in range(max_val + 1)]
        for p in range(2, max_val + 1):
            if not prime_factors[p]: # p is prime
                for multiple in range(p, max_val + 1, p):
                    prime_factors[multiple].append(p)
                    
        # Now for A and B (which are <= max_val), we have their prime factors.
        # C = 210 = [2, 3, 5, 7].
        
        def sf_product_square_free(A, B, C):
            # A, B, C are square-free integers.
            # Returns sf(A*B*C).
            # Count parity of each prime in A, B, C.
            # Since A, B, C are square-free, each prime appears at most once in each.
            # We can just XOR the sets of primes.
            
            # To make this fast, we can use a dictionary or a set.
            # Given the small size, a set is fine.
            
            # But we call this 200k times. Sets are slow.
            # Let's use a bitmask if possible?
            # Primes up to 10210. There are ~1200 primes. Too many for a single int bitmask in Python?
            # Python ints are arbitrary precision, so we can use a bitmask.
            
            # Precompute prime indices?
            # Or just use the fact that we only care about the resulting integer.
            
            # Let's try a different approach for Size 3.
            # Iterate n1, n2.
            # Compute K = sf(n1*n2*210).
            # Check if K is in restricted_kernel_map.
            
            # To compute K fast:
            # K = sf(sf(n1) * sf(n2) * 210).
            # Let A = sf(n1), B = sf(n2).
            # We need sf(A*B*210).
            # Since A, B, 210 are square-free, we can compute the product P = A*B*210.
            # Then K = sf(P).
            # P is at most 2e10.
            # We can compute sf(P) by dividing out squares of primes.
            # But we don't know the primes of P.
            
            # However, the primes of P are a subset of primes(A) U primes(B) U primes(210).
            # primes(210) = {2, 3, 5, 7}.
            # primes(A) and primes(B) are available from prime_factors[A] and prime_factors[B].
            
            # So we can get the candidate primes.
            # Then for each candidate prime p, check if p^2 divides P.
            # If so, divide P by p^2 repeatedly.
            # The remaining P is the square-free part.
            
            # This is efficient because the number of candidate primes is small (sum of number of prime factors of A, B, and 210).
            
            def sf_from_square_free_product(A, B, C):
                # A, B, C are square-free.
                # P = A * B * C
                P = A * B * C
                # Candidate primes
                cand_primes = set(prime_factors[A])
                cand_primes.update(prime_factors[B])
                cand_primes.update([2, 3, 5, 7]) # factors of 210
                
                for p in cand_primes:
                    # Remove p^2 from P
                    while P % (p * p) == 0:
                        P //= (p * p)
                return P
                
        # Now we can do the size-3 search.
        # Iterate n1, n2 in range(211, best_max) with n1 < n2.
        # Compute K = sf_from_square_free_product(sf[n1], sf[n2], 210).
        # If K in restricted_kernel_map:
        #    Find n3 in restricted_kernel_map[K] such that n3 != n1 and n3 != n2.
        #    If found, candidate_max = max(n1, n2, n3).
        #    If candidate_max < best_max, update.
            
        # To speed up, we can break early if n2 >= best_max? No, n2 < best_max by loop def.
        # But we can stop if n2 is already >= current best candidate for this n1?
        # Actually, we just want any valid n3.
        
        # Optimization: Sort restricted_nums? They are already in range order.
        # We can iterate n1 from 211 to best_max-1.
        # For each n1, iterate n2 from n1+1 to best_max-1.
        # This is O(N^2).
        
        # Let's add a time check inside the loop.
        
        for n1 in range(211, best_max):
            if time.time() > deadline:
                break
            A = sf[n1]
            for n2 in range(n1 + 1, best_max):
                if time.time() > deadline:
                    break
                B = sf[n2]
                K = sf_from_square_free_product(A, B, 210)
                if K in restricted_kernel_map:
                    # Find a valid n3
                    lst = restricted_kernel_map[K]
                    # We need n3 != n1 and n3 != n2
                    # Since n1, n2 < best_max, and lst contains numbers < best_max.
                    # We can just pick the smallest n3 in lst that is not n1 or n2.
                    # Or just check if len(lst) > 0.
                    # If n1 == n2, skip (but n1 < n2).
                    # If K == A, then n1 is in lst. We need another one.
                    # If K == B, then n2 is in lst.
                    
                    found_n3 = None
                    for n3 in lst:
                        if n3 != n1 and n3 != n2:
                            found_n3 = n3
                            break
                            
                    if found_n3 is not None:
                        candidate_max = max(n1, n2, found_n3)
                        if candidate_max < best_max:
                            best_max = candidate_max
                            best_set = [n1, n2, found_n3]
                            # We can break inner loop? No, we might find a smaller max with different n2?
                            # No, n2 is increasing. max(n1, n2, n3) >= n2.
                            # So if we found a solution, increasing n2 will only increase the max.
                            # So we can break the inner loop for this n1.
                            break
                            
    # Final check
    if best_set is None:
        return [840]
        
    # Verify
    prod = 210
    for x in best_set:
        prod *= x
    if is_square(prod):
        return [int(x) for x in best_set]
    else:
        return [840]

