import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
    N = 1000000
    max_n = N + 1
    
    # Sieve of smallest prime factors
    spf = list(range(max_n + 1))
    for i in range(2, int(max_n**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful parts using multiplicative sieve
    # P(n) = product of p^e for all prime powers p^e || n where e >= 2
    # We iterate through primes, and for each prime power p^k (k>=2),
    # we multiply p^k into P(m) for all m that are divisible by p^k but not p^(k+1).
    # Wait, that's not quite right.
    # P(n) is the product of p^e where e is the exponent of p in n, IF e >= 2.
    # So if n = p^e * m where gcd(p, m) = 1, then P(n) includes p^e if e >= 2.
    
    # A more efficient way:
    # Initialize P[n] = 1 for all n.
    # For each prime p, and for each power p^k (k >= 2),
    # for each multiple m of p^k:
    #   Let v_p(m) be the exponent of p in m.
    #   If v_p(m) >= 2, then P(m) should be multiplied by p^v_p(m).
    # But we can do this more cleverly.
    # For a fixed prime p, consider the multiples of p^2.
    # For m = k * p^2, we need to know the exact exponent of p in m.
    # Let m = p^e * r where p does not divide r.
    # We want to add p^e to the product for P(m) if e >= 2.
    
    # Alternative approach:
    # For each prime p, compute p^2, p^3, ... up to N.
    # For each power p^k (k >= 2), iterate through multiples of p^k.
    # For a multiple m of p^k, let e be the exponent of p in m.
    # We can determine e by checking divisibility by p^(k+1), etc.
    # But this might be slow if done naively.
    
    # Let's stick with the SPF-based factorization but optimize it.
    # Or, let's try the multiplicative sieve as suggested.
    
    P = [1] * (max_n + 1)
    
    # Generate primes from spf
    primes = []
    for i in range(2, max_n + 1):
        if spf[i] == i:
            primes.append(i)
    
    for p in primes:
        # For each power p^k with k >= 2
        power = p * p
        while power <= max_n:
            # For each multiple m of power
            # We need to find the exact exponent of p in m.
            # Let m = power * t.
            # The exponent of p in m is k + v_p(t).
            # We want to multiply P[m] by p^(k + v_p(t)).
            # But we already might have multiplied by some part?
            # No, we process each prime independently.
            # So for this prime p, we will multiply P[m] by p^(v_p(m)) if v_p(m) >= 2.
            
            # To avoid double counting or incorrect multiplication,
            # we should only multiply by the FULL p^e part.
            # Since we are iterating primes, each prime's contribution is independent.
            
            # For each multiple m of p^k:
            #   e = v_p(m)
            #   P[m] *= p^e
            
            # How to compute e efficiently?
            # We can check if m is divisible by p^(k+1), p^(k+2), etc.
            # But this is slow.
            
            # Better:
            # For a fixed p, iterate m from p^2 to N in steps of p^2?
            # No, that misses multiples with higher powers.
            
            # Let's iterate m from 1 to N.
            # If m is divisible by p^2, then v_p(m) >= 2.
            # We can compute v_p(m) by dividing m by p until it's not divisible.
            # But doing this for every m and every p is O(N log N) or worse.
            
            # Let's try a different approach.
            # For each prime p, for each multiple m of p^2:
            #   Let temp = m
            #   e = 0
            #   while temp % p == 0:
            #       temp //= p
            #       e += 1
            #   P[m] *= p^e
            
            # This is still potentially slow.
            
            # Let's go back to the SPF method but precompute P[n] for all n.
            # Factorization using SPF is O(log n) per number.
            # Total O(N log N) which is ~14e6 operations. Should be fast enough.
            
            break # Stop the power loop, we'll do it differently
    
    # Let's use the SPF to precompute P[n]
    for n in range(2, max_n + 1):
        original_n = n
        result = 1
        while n > 1:
            p = spf[n]
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count >= 2:
                result *= (p ** count)
        P[original_n] = result
    
    best_n = 10000
    max_prod = 0
    
    for n in range(10000, N):
        if time.time() > deadline:
            break
            
        prod = P[n] * P[n + 1]
        if prod > max_prod:
            max_prod = prod
            best_n = n
            
    return best_n
