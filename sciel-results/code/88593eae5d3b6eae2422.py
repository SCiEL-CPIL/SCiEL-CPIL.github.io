import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer before 120s limit
    
    # Powerful numbers of the form R * (u^2 + v^2)
    # Let N = R * M
    # R is powerful. M is powerful? No, M = u^2 + v^2.
    # For N to be a sum of two powerful numbers a+b=N:
    # Let a = R * u^2 and b = R * v^2.
    # a is powerful iff R*u^2 is powerful.
    # b is powerful iff R*v^2 is powerful.
    # If R is powerful, then R*u^2 is powerful iff u is powerful? 
    # No. Let R = \prod p_i^{e_i} with e_i >= 2.
    # u = \prod p_i^{f_i} \prod q_j^{g_j}.
    # R*u^2 = \prod p_i^{e_i + 2f_i} \prod q_j^{2g_j}.
    # This is powerful iff for all prime factors, exponent >= 2.
    # For p_i, e_i + 2f_i >= 2 is always true if e_i>=2.
    # For q_j, 2g_j >= 2 => g_j >= 1. So u must be squarefree? No.
    # If u has a prime factor q with exponent 1, then in R*u^2 it has exponent 2. Powerful.
    # If u has a prime factor q with exponent 0, it's not in u.
    # So u can be any integer? 
    # Wait. If u is not powerful, does R*u^2 become powerful?
    # Example: R=4 (powerful), u=2 (powerful). R*u^2 = 4*4=16 (powerful).
    # Example: R=4, u=3. R*u^2 = 4*9=36=2^2*3^2 (powerful).
    # Example: R=4, u=6. R*u^2 = 4*36=144=2^4*3^2 (powerful).
    # It seems if R is powerful, R*u^2 is ALWAYS powerful.
    # Proof: Let p be a prime factor of R*u^2.
    # Case 1: p divides R. Then p^2 divides R (since R powerful). So p^2 divides R*u^2.
    # Case 2: p does not divide R. Then p divides u^2, so p divides u. Thus p^2 divides u^2, so p^2 divides R*u^2.
    # Therefore, if R is powerful, then for ANY integer u, R*u^2 is powerful.
    
    # So the strategy is:
    # 1. Pick a powerful number R.
    # 2. Pick an integer M.
    # 3. Find all pairs (u, v) such that u^2 + v^2 = M.
    # 4. Then a = R*u^2 and b = R*v^2 are powerful and a+b = R*(u^2+v^2) = R*M = N.
    # 5. We need N <= 10^18.
    # 6. The number of pairs is the number of representations of M as sum of two squares.
    
    # To maximize the number of representations of M as sum of two squares, 
    # M should have many prime factors congruent to 1 mod 4.
    # The number of representations r_2(M) is 4 * (d_1(M) - d_3(M)), where d_1 is number of divisors 1 mod 4 and d_3 is 3 mod 4.
    # If M = 2^k * \prod p_i^{a_i} * \prod q_j^{b_j} where p_i = 1 mod 4, q_j = 3 mod 4.
    # If any b_j is odd, r_2(M) = 0.
    # If all b_j are even, r_2(M) = 4 * \prod (a_i + 1).
    # We want to maximize \prod (a_i + 1) subject to M <= 10^18 / R.
    
    # Let's choose R=1 (powerful). Then N=M.
    # We need to find M <= 10^18 with many prime factors 1 mod 4, all with even exponents?
    # Wait, the formula r_2(M) = 4 * \prod (a_i + 1) applies if M is of the form 2^k * \prod p_i^{a_i} * \prod q_j^{2c_j}.
    # Actually, the standard formula is:
    # If M = 2^k * \prod_{i=1}^m p_i^{a_i} * \prod_{j=1}^n q_j^{b_j}
    # where p_i are primes 1 mod 4, q_j are primes 3 mod 4.
    # Then r_2(M) = 0 if any b_j is odd.
    # Otherwise, r_2(M) = 4 * \prod_{i=1}^m (a_i + 1).
    
    # So we want to maximize \prod (a_i + 1) for M <= 10^18, with M having no prime factors 3 mod 4 to an odd power.
    # To maximize the product, we should use small primes 1 mod 4: 5, 13, 17, 29, 37, 41, ...
    
    # Let's search for M of the form \prod p_i^{a_i} where p_i are primes 1 mod 4.
    # We can also include factors of 2. 2 does not affect the count (it just rotates/scals).
    # Actually, 2 = 1^2 + 1^2.
    # If M = 2^k * M', r_2(M) = r_2(M').
    # So we can ignore factors of 2 for the count, but they help us fit M <= 10^18 if M' is too big? No, we want M as large as possible?
    # No, we want to maximize the count. The count depends only on the exponents of primes 1 mod 4.
    # So we should use the smallest primes 1 mod 4 to get the highest exponents.
    
    primes_1_mod_4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 997]
    
    best_count = 0
    best_M = 1
    best_exponents = []
    
    # DFS to find M maximizing prod(a_i+1)
    # Limit M <= 10^18
    # We can also multiply by R (powerful) to get N.
    # But R=1 is simplest. If R>1, we have less room for M.
    # So R=1 is optimal for maximizing the count of representations of M.
    
    def dfs(idx, current_M, current_count, exponents):
        nonlocal best_count, best_M, best_exponents
        if time.time() > deadline:
            return
        
        if current_count > best_count:
            best_count = current_count
            best_M = current_M
            best_exponents = exponents[:]
        
        if idx >= len(primes_1_mod_4):
            return
        
        p = primes_1_mod_4[idx]
        
        # Try all possible exponents for p
        # p^a <= 10^18 / current_M
        max_a = 0
        while (p ** (max_a + 1)) * current_M <= 10**18:
            max_a += 1
        
        for a in range(max_a, -1, -1):
            if a == 0:
                # Prune: if we skip this prime, can we still beat best_count?
                # Upper bound on remaining count
                # This is hard to bound tightly, so just proceed.
                pass
            
            new_M = current_M * (p ** a)
            new_count = current_count * (a + 1)
            exponents.append(a)
            dfs(idx + 1, new_M, new_count, exponents)
            exponents.pop()
            
            # Optimization: if p is large, a will be small.
            # If a=0, we can break if p^2 > 10^18 / current_M? No.
            # But if p is too large to fit even a=1, we can break.
            if p > 10**18 // current_M:
                break
    
    dfs(0, 1, 1, [])
    
    # Now we have best_M and best_exponents.
    # N = best_M.
    # We need to find all pairs (u, v) such that u^2 + v^2 = best_M.
    # The number of such pairs is r_2(best_M) / 4 (since u,v can be negative and swapped).
    # We need to list them.
    
    # To generate the pairs, we can use the Gaussian integers or a simple search.
    # Since best_M can be up to 10^18, a simple search u=0..sqrt(M) is too slow (10^9).
    # We need a faster way to generate the pairs.
    
    # Alternative: Use the factorization in Gaussian integers.
    # Or, since we know the count, we can just return the count?
    # The task requires returning the pairs [[a, b], ...].
    # So we MUST generate the pairs.
    
    # Generating pairs for M up to 10^18 is hard if M is large.
    # But the count is small (max product of (a_i+1)).
    # For M=5^10, count = 11.
    # For M=5^5 * 13^3 * 17^2, count = 6*4*3 = 72.
    # The number of pairs is manageable (hundreds).
    
    # How to generate them?
    # We can use the identity:
    # (u1^2+v1^2)(u2^2+v2^2) = (u1u2 - v1v2)^2 + (u1v2 + v1u2)^2
    # We can start with the prime factors and build up the representations.
    
    # Representations of p (prime 1 mod 4) as sum of two squares:
    # Find x, y such that x^2 + y^2 = p.
    # This can be done by searching x=1..sqrt(p).
    
    def sum_two_squares_prime(p):
        for x in range(1, isqrt(p) + 1):
            y2 = p - x * x
            y = isqrt(y2)
            if y * y == y2:
                return (x, y)
        return (1, 0) # Should not happen for p=1 mod 4
    
    # Combine representations
    # If we have lists of pairs for each prime power, we can combine them.
    # But it's easier to just generate all pairs for M directly if M is not too large?
    # No, M can be 10^18.
    
    # Let's use the recursive combination.
    # Start with [(1, 0)] for M=1.
    # For each prime factor p^a, we need to combine the current list with the representations of p^a.
    # Representations of p^a:
    # If p = x^2 + y^2, then p^a can be represented as (x+iy)^a in Gaussian integers.
    # We can generate all a+1 representations of p^a.
    
    def gaussian_mul(z1, z2):
        (a, b), (c, d) = z1, z2
        return (a*c - b*d, a*d + b*c)
    
    def gaussian_pow(z, n):
        result = (1, 0)
        base = z
        while n:
            if n % 2:
                result = gaussian_mul(result, base)
            base = gaussian_mul(base, base)
            n //= 2
        return result
    
    # For a prime p = x^2 + y^2, the representations of p^a are given by the real and imaginary parts of (x+iy)^k for k=0..a?
    # No. The number of representations is a+1.
    # They correspond to the divisors in Gaussian integers.
    # Actually, the representations of p^a are (Re((x+iy)^k * (x-iy)^{a-k}), Im(...)) for k=0..a?
    # No, that's not quite right.
    
    # Simpler: Just find all (u, v) such that u^2+v^2=M by iterating u?
    # If M is large, this is slow.
    # But the count is small.
    # Is there a way to find them quickly?
    
    # Let's try a different approach for generating pairs.
    # We know the count. Let's just return a dummy list if we can't generate them quickly?
    # No, the evaluator checks if a+b=N and a,b powerful.
    
    # Let's optimize the generation.
    # If M is a product of small primes, we can generate the pairs recursively.
    
    pairs_uv = [(1, 0)]
    
    for i, p in enumerate(primes_1_mod_4):
        a = best_exponents[i] if i < len(best_exponents) else 0
        if a == 0:
            continue
        
        # Find representation of p
        x, y = sum_two_squares_prime(p)
        z = (x, y)
        
        # Generate representations of p^a
        # The representations of p^a are given by the coefficients of (x+iy)^a?
        # No.
        # Let's just collect all pairs for p^a by combining p repeatedly.
        # Reps of p: [(x, y), (x, -y), (-x, y), ...] -> essentially (x, y) up to symmetry.
        # We can just use the Gaussian integer multiplication.
        
        new_pairs = []
        for (u, v) in pairs_uv:
            # Multiply (u, v) by all representations of p^a?
            # No, we need to combine with the representations of p^a.
            # The representations of p^a are the set of (Re(z^k * w), Im(z^k * w)) where w is a unit?
            # Actually, the number of representations of p^a is a+1.
            # They are generated by (x+iy)^k * (x-iy)^{a-k} for k=0..a.
            # Let's verify for a=1:
            # k=0: (x-iy) -> (x, -y)
            # k=1: (x+iy) -> (x, y)
            # These are the two essentially distinct ones (up to sign/symmetry).
            
            # For each k in 0..a:
            for k in range(a + 1):
                # Compute z1 = (x+iy)^k
                # Compute z2 = (x-iy)^(a-k)
                # Product = z1 * z2
                # This gives one representation.
                
                # Optimization: precompute powers of z and conj(z)
                # This might be slow if done inside the loop.
                
                pass
        
        # This is getting complicated. Let's try a simpler heuristic.
        # If M is small enough (< 10^12), we can iterate u.
        # If M is large, we might have few representations, but finding them is hard.
        
        # Given the time limit, let's just return the count and a few valid pairs if we can find them.
        # But we need ALL pairs to maximize the score?
        # "The score is the number of pairs".
        # So we need to list them all.
        
        break # Exit loop, handle generation separately
    
    # Let's try to generate pairs for best_M.
    # If best_M is too large to iterate, we are stuck.
    # But best_M is composed of small primes, so it might be large.
    
    # Let's use the recursive Gaussian integer method properly.
    
    def get_reps_prime_power(p, a):
        """Return list of (u, v) such that u^2+v^2=p^a, u>=0, v>=0, u<=v."""
        x, y = sum_two_squares_prime(p)
        z = (x, y)
        conj_z = (x, -y)
        
        reps = set()
        # Precompute powers
        powers_z = [(1, 0)]
        powers_conj = [(1, 0)]
        for i in range(1, a + 1):
            powers_z.append(gaussian_mul(powers_z[-1], z))
            powers_conj.append(gaussian_mul(powers_conj[-1], conj_z))
        
        for k in range(a + 1):
            # z^k * conj_z^(a-k)
            prod = gaussian_mul(powers_z[k], powers_conj[a - k])
            u, v = prod
            u, v = abs(u), abs(v)
            if u > v:
                u, v = v, u
            reps.add((u, v))
        
        return list(reps)
    
    all_pairs = [(1, 0)]
    
    for i, p in enumerate(primes_1_mod_4):
        a = best_exponents[i] if i < len(best_exponents) else 0
        if a == 0:
            continue
        
        reps = get_reps_prime_power(p, a)
        
        new_all_pairs = []
        for (u1, v1) in all_pairs:
            for (u2, v2) in reps:
                # Combine (u1, v1) and (u2, v2)
                # (u1+iv1)(u2+iv2) and (u1+iv1)(u2-iv2)
                # We need to add all combinations.
                # Actually, the set of representations of the product is the closure under the Brahmagupta identity.
                # It's easier to just multiply the Gaussian integers.
                
                # (u1, v1) represents a factor.
                # (u2, v2) represents a factor.
                # The product represents the product.
                # There are 4 symmetries.
                
                prod1 = gaussian_mul((u1, v1), (u2, v2))
                prod2 = gaussian_mul((u1, v1), (u2, -v2))
                
                for prod in [prod1, prod2]:
                    u, v = prod
                    u, v = abs(u), abs(v)
                    if u > v:
                        u, v = v, u
                    new_all_pairs.append((u, v))
        
        # Deduplicate
        all_pairs = list(set(new_all_pairs))
        
        if time.time() > deadline:
            break
    
    # Now all_pairs contains (u, v) such that u^2+v^2=best_M.
    # We need to form a = R*u^2, b = R*v^2.
    # R=1.
    # So a=u^2, b=v^2.
    
    final_pairs = []
    for u, v in all_pairs:
        a = u * u
        b = v * v
        if a + b == best_M:
            if a <= b:
                final_pairs.append([a, b])
    
    # Deduplicate final pairs
    final_pairs = [list(p) for p in set(tuple(sorted(p)) for p in final_pairs)]
    
    if final_pairs:
        return {'N': best_M, 'pairs': final_pairs}
    else:
        return {'N': 2, 'pairs': [[1, 1]]}
