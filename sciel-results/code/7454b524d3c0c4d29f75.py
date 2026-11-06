import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s kill

    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Generate primes up to limit using sieve
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_prime_arr = bytearray(b'\x01') * (limit + 1)
        is_prime_arr[0] = is_prime_arr[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime_arr[i]:
                is_prime_arr[i*i::i] = bytearray(len(is_prime_arr[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime_arr[i]]

    # Precompute small primes for trial division
    # We need primes up to sqrt(max_number) where max_number is roughly p^2 + 200
    # If p is up to 10^7, p^2 is 10^14, sqrt is 10^7.
    # But we only need to factor out primes <= p.
    # For the segmented sieve approach, we need all primes <= p.
    # So we need a list of primes up to the maximum p we search.

    # Strategy:
    # 1. Generate primes up to P_MAX (e.g., 5,000,000 or 10,000,000 depending on time).
    # 2. For each prime p in this list:
    #    - Use a segmented sieve on the interval [p^2+1, p^2+200] using all primes <= p.
    #    - This allows us to check all 200 numbers simultaneously for p-smoothness.
    #    - If a number has a remaining factor > 1 after dividing by all primes <= p, it's not p-smooth.
    #    - Count the run length.

    # Let's estimate time for segmented sieve.
    # For a given p, we have 200 numbers.
    # We iterate through all primes q <= p.
    # For each q, we check which of the 200 numbers are divisible by q.
    # This is roughly O(200 * pi(p)) operations? No, that's too slow.
    # 
    # Better segmented sieve:
    # Create an array `rem` of size 200, initialized to [p^2+1, ..., p^2+200].
    # For each prime q <= p:
    #    Find the smallest index i in [0, 199] such that (p^2 + 1 + i) % q == 0.
    #    If it exists, divide rem[i], rem[i+q], rem[i+2q], ... by q until they are no longer divisible.
    #    Actually, we just need to know if rem[i] becomes 1.
    #    We can stop processing q if q > max(rem).
    
    # The cost per p is sum over q<=p of (200/q) divisions.
    # Sum_{q<=p} 200/q ≈ 200 * log(log(p)).
    # For p=10^7, log(log(10^7)) ≈ log(16) ≈ 2.8. So ~560 divisions per prime.
    # Number of primes up to 10^7 is ~664,000.
    # Total operations: 664,000 * 560 ≈ 370 million simple ops.
    # In Python, this might be too slow. 370M ops / 10^7 ops/sec = 37 seconds.
    # This is feasible within 135s if optimized.

    # Let's try to search up to P_MAX = 5,000,000 first to be safe.
    # pi(5e6) ≈ 348,000.
    # 348,000 * 500 ≈ 174M ops. Should take ~15-20s.
    # Then we have time for random larger primes.

    P_MAX = 5000000
    
    # Generate primes up to P_MAX
    primes = sieve_primes(P_MAX)
    
    best_k = 0
    best_p = 2

    # Precompute logs for quick check? No, just use the sieve.
    
    for idx, p in enumerate(primes):
        if time.time() > deadline - 5:
            break
            
        # Segmented sieve for [p^2+1, p^2+200]
        # rem[i] will hold the remaining part of p^2+1+i after removing all prime factors <= p
        # If rem[i] == 1, it is p-smooth.
        # We need to divide by all primes q <= p.
        
        # Optimization: We only need to divide by primes q <= sqrt(max(rem)).
        # Initially max(rem) is p^2+200. sqrt is approx p.
        # So we need all primes q <= p.
        
        # To speed up, we can stop the inner loop early if q*q > max(rem).
        # But max(rem) decreases.
        
        rem = [p*p + 1 + i for i in range(200)]
        
        # We iterate through primes q.
        # We can break if q > p.
        # We can also break if q*q > max(rem).
        
        # To find start index for q:
        # We need i such that (p^2 + 1 + i) % q == 0
        # i = (- (p^2 + 1)) % q
        # But we only care about i in [0, 199].
        
        for q in primes:
            if q > p:
                break
            # Check if q*q > max(rem). If so, any remaining factor is prime.
            # But we still need to divide by q if it divides any rem[i].
            # Actually, if q > sqrt(max(rem)), then q can divide rem[i] at most once.
            # And if q divides rem[i], then rem[i]/q < sqrt(max(rem)) < q.
            # So rem[i]/q must be 1 or a prime < q.
            # Since we've divided by all primes < q, rem[i]/q must be 1.
            # So if q > sqrt(max(rem)), we just check divisibility and divide by q once.
            
            # Calculate start index
            start = (-(p*p + 1)) % q
            if start >= 200:
                # No multiple in range
                # But wait, start is the offset from p^2+1.
                # If start >= 200, then the first multiple is p^2+1+start > p^2+200.
                # So no multiples.
                continue
                
            # Iterate through indices start, start+q, ...
            # We need to divide rem[i] by q as many times as possible.
            
            max_rem = max(rem)
            if q * q > max_rem:
                # Fast path: just divide once if divisible
                for i in range(start, 200, q):
                    if rem[i] % q == 0:
                        rem[i] //= q
            else:
                for i in range(start, 200, q):
                    while rem[i] % q == 0:
                        rem[i] //= q
                        
        # Count the run
        # A number is p-smooth if rem[i] == 1
        # We need the longest prefix of 1s.
        k = 0
        for i in range(200):
            if rem[i] == 1:
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                return {'p': p, 'k': 200}

    # If we haven't found 200, try some random larger primes
    # For larger p, we can't use the full sieve of primes up to p if p is huge.
    # But we can use the primes we already generated (up to P_MAX) to strip small factors.
    # Then use Pollard-Rho for the rest.
    
    # Let's try a few random primes in [P_MAX, 10^9]
    # We need to generate them.
    
    # Helper: find next prime
    def next_prime(n):
        if n <= 2:
            return 2
        if n % 2 == 0:
            n += 1
        else:
            n += 2
        while not is_prime_mr(n):
            n += 2
        return n

    # Try 100 random primes
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        p = random.randint(P_MAX + 100, 10**9)
        p = next_prime(p)
        
        # Check run for this p
        # We will strip factors using primes up to P_MAX (or smaller if p < P_MAX, but p > P_MAX here)
        # Then use Pollard-Rho for remaining.
        
        # Actually, for p > P_MAX, we only have primes up to P_MAX.
        # If p is much larger than P_MAX, we might miss factors between P_MAX and p.
        # So we must use Pollard-Rho to fully factorize or determine if there's a factor > p.
        
        # Strategy for large p:
        # For each n in p^2+1 .. p^2+200:
        # 1. Divide by all primes q <= P_MAX.
        # 2. If rem == 1, it's p-smooth (since p > P_MAX, all factors are <= P_MAX < p).
        # 3. If rem > 1:
        #    - If rem is prime, check if rem <= p.
        #    - If rem is composite, factor it. If any factor > p, fail.
        
        k = 0
        for i in range(1, 201):
            n = p*p + i
            rem = n
            for q in primes:
                if q > p: # Should not happen since p > P_MAX
                    break
                if q*q > rem:
                    break
                while rem % q == 0:
                    rem //= q
            if rem == 1:
                k += 1
            else:
                # Check if rem is prime
                if is_prime_mr(rem):
                    if rem <= p:
                        k += 1
                    else:
                        break
                else:
                    # Factor rem
                    # If any factor > p, break
                    # Simple Pollard Rho
                    # We need a robust factorizer
                    def pollard_rho(n):
                        if n % 2 == 0: return 2
                        if n % 3 == 0: return 3
                        while True:
                            c = random.randrange(1, n)
                            x = random.randrange(2, n)
                            y = x
                            d = 1
                            while d == 1:
                                x = (x*x + c) % n
                                y = (y*y + c) % n
                                y = (y*y + c) % n
                                d = math.gcd(abs(x-y), n)
                            if d != n:
                                return d
                    def factor(n):
                        if n == 1: return []
                        if is_prime_mr(n): return [n]
                        d = pollard_rho(n)
                        return factor(d) + factor(n//d)
                    
                    factors = factor(rem)
                    if all(f <= p for f in factors):
                        k += 1
                    else:
                        break
                        
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                return {'p': p, 'k': 200}

    return {'p': best_p, 'k': best_k}
