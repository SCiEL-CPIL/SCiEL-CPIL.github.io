import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # 1. Generate all powerful numbers up to N_MAX
    # A number is powerful if all prime factors have exponent >= 2
    # We can generate them using a sieve-like approach or recursive generation.
    # Recursive generation is efficient for powerful numbers because there are few of them.
    # Count of powerful numbers up to 10^6 is roughly ~1000-2000.
    
    powerful_numbers = []
    
    def gen_powerful(current, min_prime, max_val):
        if current > max_val:
            return
        if current >= 1:
            powerful_numbers.append(current)
        # Try multiplying by p^2, p^3, ... for primes p >= min_prime
        # To avoid duplicates, we enforce p >= min_prime
        p = min_prime
        while p * p <= max_val // current:
            power = p * p
            while current * power <= max_val:
                gen_powerful(current * power, p, max_val)
                power *= p
            p += 1
            # Find next prime
            # This is inefficient if we just increment p.
            # Better to precompute primes or use a prime generator.
            
    # Let's use a better generation method.
    # Precompute primes up to sqrt(N_MAX) or N_MAX?
    # Smallest prime factor sieve.
    
    # Actually, let's just use the sieve method from the original but optimized, 
    # or better, just generate powerful numbers directly.
    
    # List of primes up to N_MAX
    # Simple sieve for primes
    is_prime = [True] * (int(N_MAX**0.5) + 1)
    # Wait, we need primes up to N_MAX? No, for powerful numbers, 
    # the smallest prime factor squared must be <= N_MAX.
    # So primes up to sqrt(N_MAX) are sufficient for the "base" of the power,
    # but we can have higher powers.
    # Actually, if p > sqrt(N_MAX), then p^2 > N_MAX, so p cannot be a factor 
    # with exponent >= 2 in a number <= N_MAX.
    # So we only need primes up to sqrt(N_MAX).
    
    limit = int(N_MAX**0.5) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    
    powerful_numbers = [1]
    
    def add_powerful(base, min_prime_idx, max_val):
        # Append base to powerful_numbers if not already there? 
        # We generate uniquely by enforcing min_prime_idx.
        if base > max_val:
            return
        powerful_numbers.append(base)
        for i in range(min_prime_idx, len(primes)):
            p = primes[i]
            if p * p > max_val // base:
                break
            power = p * p
            while base * power <= max_val:
                add_powerful(base * power, i, max_val)
                power *= p
                
    add_powerful(1, 0, N_MAX)
    
    # Remove duplicates? The generation should be unique if min_prime_idx is handled correctly.
    # Let's sort and unique just in case.
    powerful_numbers = sorted(list(set(powerful_numbers)))
    
    # 2. Search for best A, B
    # We want to maximize A * B
    # Such that gcd(A, B) == 1
    # And there exists n in [N_MIN, N_MAX] with n % A == 0 and n % B == -1
    
    best_val = 0
    best_n = N_MIN
    
    # To speed up, we can iterate A and B.
    # Since we want to maximize A*B, we can sort powerful numbers in descending order
    # and prune.
    # However, the number of powerful numbers is small (~1000).
    # 1000 * 1000 = 1,000,000 pairs.
    # For each pair, we check gcd and CRT.
    # This should be fast enough in Python if optimized.
    
    pn = powerful_numbers
    n_pn = len(pn)
    
    # We can filter out 1? 
    # If A=1, P(n) >= 1. If B=1, P(n+1) >= 1.
    # But we want to maximize product.
    
    for i in range(n_pn):
        A = pn[i]
        if A > N_MAX:
            continue
        for j in range(n_pn):
            B = pn[j]
            if B > N_MAX:
                continue
            
            # Pruning: if A * B <= best_val, we might still find a larger one?
            # No, we are maximizing A*B. 
            # If we sort in descending order, we can break early?
            # Let's sort descending.
            
            if math.gcd(A, B) != 1:
                continue
                
            # Solve n = k*A = m*B - 1
            # k*A + 1 = m*B
            # We need n such that n % A == 0 and n % B == -1
            # This is equivalent to n % A == 0 and n % B == B-1
            
            # CRT:
            # n = A * x
            # A * x = -1 (mod B)
            # x = -A^{-1} (mod B)
            # Since gcd(A, B) = 1, A^{-1} exists.
            
            try:
                inv_A = pow(A, -1, B)
            except ValueError:
                continue # Should not happen if gcd is 1
                
            x = (-inv_A) % B
            
            # n = A * x is the smallest non-negative solution modulo L = A*B
            n0 = A * x
            
            # General solution: n = n0 + k * (A * B)
            L = A * B
            
            # Find if any n in [N_MIN, N_MAX]
            # n_min = smallest n >= N_MIN of form n0 + k*L
            # n_max = largest n <= N_MAX of form n0 + k*L
            
            # k_min = ceil((N_MIN - n0) / L)
            # k_max = floor((N_MAX - n0) / L)
            
            if n0 >= N_MIN:
                k_min = 0
            else:
                k_min = (N_MIN - n0 + L - 1) // L
                
            k_max = (N_MAX - n0) // L
            
            if k_min <= k_max:
                # There is a valid n
                # The powerful part of n is at least A.
                # The powerful part of n+1 is at least B.
                # So product is at least A * B.
                # It could be larger if n has other prime factors with exponent >= 2.
                # But A*B is a lower bound.
                # We want to maximize the *actual* powerful part product.
                # However, A and B are powerful numbers dividing n and n+1.
                # So P(n) is a multiple of A? 
                # Yes, if A = prod p^e, e>=2, and A|n, then n has p^e, so P(n) includes p^e.
                # So P(n) >= A.
                # Similarly P(n+1) >= B.
                # So P(n)P(n+1) >= A*B.
                
                # To be precise, we should calculate P(n) and P(n+1) for the candidate n.
                # But calculating P(n) requires factorization.
                # We can just use A*B as the score for pruning, and then verify with actual P(n)P(n+1) for the top candidates?
                # Or just assume A*B is the value?
                # The task says "Maximize powerful_part(n)*powerful_part(n+1)".
                # If we find n via CRT with A, B, the actual powerful parts might be larger.
                # Example: n = 12 = 2^2 * 3. P(n) = 4. A=4.
                # n+1 = 13. P(n+1) = 1. B=1.
                # Product = 4. A*B = 4.
                # Example: n = 8 = 2^3. P(n) = 8. A=4 (divides 8).
                # If we picked A=4, B=1, we get score 4.
                # But actual P(n) is 8.
                # So A*B is a lower bound.
                
                # We should calculate the actual value for the best candidates.
                # For now, let's track the best A*B and the corresponding n.
                # Then at the end, we can evaluate P(n)P(n+1) for the top few.
                
                val = A * B
                if val > best_val:
                    best_val = val
                    # Pick the smallest n in the range to be deterministic?
                    # Or any n.
                    best_n = n0 + k_min * L
                    
    # Verify the best candidate and maybe check neighbors or other high A*B pairs?
    # Actually, the "lower bound" A*B might miss the true maximum if the true maximum comes from a pair (A,B) where A*B is smaller but P(n)P(n+1) is larger due to extra factors.
    # However, if n has extra powerful factors, those factors are also powerful numbers dividing n.
    # So we could have chosen a larger A.
    # Example: n=8. Powerful divisors: 1, 4, 8.
    # If we iterate A=8, B=1, we get A*B=8.
    # If we iterate A=4, B=1, we get A*B=4.
    # So the maximum A*B over all valid pairs (A,B) should correspond to the maximum possible lower bound.
    # And since P(n) is the *product of all* powerful prime powers, P(n) is exactly the largest powerful divisor of n.
    # So if we let A = P(n) and B = P(n+1), then A divides n and B divides n+1.
    # And A and B are powerful.
    # So the pair (P(n), P(n+1)) is a valid pair in our search space.
    # And for this pair, A*B = P(n)P(n+1).
    # So maximizing A*B over all coprime powerful pairs (A,B) that allow a solution n in the range
    # is EXACTLY equivalent to maximizing P(n)P(n+1).
    
    # Wait, is it guaranteed that P(n) and P(n+1) are coprime?
    # P(n) divides n. P(n+1) divides n+1.
    # gcd(n, n+1) = 1.
    # So any divisor of n and any divisor of n+1 are coprime.
    # So gcd(P(n), P(n+1)) = 1.
    # Yes, they are coprime.
    
    # So the algorithm is correct.
    
    # To ensure we don't miss anything, we should probably check the actual P(n)P(n+1) for the best_n found.
    # But since we argued that max A*B == max P(n)P(n+1), best_n should be optimal.
    
    # One edge case: What if the best A*B pair yields multiple n's?
    # We picked the smallest n. Does it matter?
    # P(n) depends on n.
    # If n = n0 + k*L, does P(n) change?
    # n is a multiple of A. So P(n) is a multiple of A.
    # n+1 is a multiple of B. So P(n+1) is a multiple of B.
    # It is possible that for some k, P(n) is larger than A?
    # Yes, if n has other prime factors with exponent >= 2.
    # But if so, let A' be that larger powerful divisor.
    # Then A' is also a powerful number.
    # And A' divides n, so A' divides n0 + kL.
    # Since A divides n0 and A divides L, A divides n.
    # If A' divides n, then A' is a powerful divisor of n.
    # In our search, we would have considered the pair (A', B') where B' is the corresponding powerful part of n+1.
    # And A'*B' >= A*B.
    # So the maximum A*B found in the search will correspond to the true maximum P(n)P(n+1).
    
    return best_n
