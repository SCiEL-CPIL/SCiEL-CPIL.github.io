import time
import math
import numpy as np

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    # 1. Sieve primes up to B
    # We need primes up to sqrt(max_N).
    # If p goes up to 10^7, p^2 is 10^14. sqrt is 10^7.
    # Sieve up to 10^7 is feasible.
    B = 10**7
    sieve = np.ones(B + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(math.isqrt(B)) + 1):
        if sieve[i]:
            sieve[i*i:B+1:i] = False
    primes = np.array(np.nonzero(sieve))[0]
    primes_list = primes.tolist()
    
    # 2. Smoothness Check
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        if n <= p:
            return True
            
        # We need to divide by primes q <= min(p, sqrt(n))
        # If n shrinks, sqrt(n) shrinks.
        # We iterate through primes_list.
        # primes_list is sorted.
        
        # Optimization: if p is small, limit is p.
        # If p is large, limit is sqrt(n).
        
        # We can break early if q*q > n and q > p? No.
        # Condition to stop trial division: q > min(p, sqrt(n))
        
        limit = min(p, math.isqrt(n))
        
        # Iterate primes
        # Using list iteration is faster than numpy for small loops?
        # Or just use the list directly.
        
        for q in primes_list:
            if q > limit:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                # Update limit
                new_limit = min(p, math.isqrt(n))
                if new_limit < limit:
                    limit = new_limit
                    # If limit dropped below current q, we might need to break?
                    # But q is increasing. If limit < q, we break next iteration.
        
        if n == 1:
            return True
            
        # n > 1
        # We stopped because q > limit.
        # Two cases:
        # 1. limit == p (i.e., p <= sqrt(original_n))
        #    Then we divided by all primes <= p.
        #    n has no prime factor <= p.
        #    So n is not p-smooth.
        # 2. limit < p (i.e., p > sqrt(original_n))
        #    Then we divided by all primes <= sqrt(original_n).
        #    n is prime.
        #    Check if n <= p.
        
        # How to distinguish?
        # If p <= math.isqrt(original_n):
        #    return False
        # Else:
        #    return n <= p
        
        # We didn't store original_n.
        # But if p <= limit_initial, we are in case 1.
        # limit_initial = min(p, isqrt(n_initial))
        # If p <= isqrt(n_initial), limit_initial = p.
        # If p > isqrt(n_initial), limit_initial = isqrt(n_initial).
        
        # We can just check:
        # If the loop stopped because q > p, then n is not smooth (if n>1).
        # If the loop stopped because q > isqrt(n), then n is prime.
        
        # Since we update limit, the stopping condition is q > limit.
        # If limit was p, and we stopped, n has no factor <= p. Return False.
        # If limit was isqrt(n), and we stopped, n is prime. Return n <= p.
        
        # We can infer from the final value of limit?
        # No, limit changes.
        
        # Let's just check:
        # If p <= math.isqrt(n + 1): # Rough check
        #    return False
        # Else:
        #    return n <= p
        
        # More robust:
        # If we divided by all primes <= p, and n > 1, then n has a factor > p.
        # This happens if the loop ran until q > p.
        # If the loop ran until q > isqrt(n), then n is prime.
        
        # We can track if we hit the p-bound or the sqrt-bound.
        # But simpler:
        # If n is prime, it's smooth if n <= p.
        # If n is composite, it's not smooth (because we removed all small factors).
        # So: is n prime?
        # If yes, return n <= p.
        # If no, return False.
        
        # But is_prime is expensive.
        # However, if p <= isqrt(n_initial), we know n is not smooth without is_prime.
        # So:
        if p <= math.isqrt(n + 1): # n is current remainder, but we need original?
            # Actually, if p <= isqrt(original_n), then we divided by all primes <= p.
            # Any remaining factor is > p.
            # So return False.
            # How to know if p <= isqrt(original_n)?
            # We can just check if p <= math.isqrt(n_initial).
            # We don't have n_initial.
            
            # Let's pass n_initial or just use the logic:
            # If the loop stopped because q > p, return False.
            # If the loop stopped because q > isqrt(n), return n <= p.
            
            # We can just check:
            # If math.isqrt(n) < p:
            #    # We stopped because of sqrt limit?
            #    # Not necessarily.
            
            # Let's use a flag.
            pass
        
        # Simplest robust method:
        # If n > 1:
        #   If n is prime: return n <= p
        #   Else: return False
        
        # But we can avoid is_prime if we know we exhausted primes up to p.
        # If p <= math.isqrt(n_initial):
        #    return False
        # Else:
        #    return n <= p # n is prime
        
        # We need n_initial.
        # Let's modify the function to take n_initial? No, n changes.
        # Let's just use is_prime for the remainder if we are in the "large p" regime.
        # But is_prime is fast for small numbers.
        
        # Heuristic:
        # If p < 10**6, we are likely in the p-bound regime.
        # If p > 10**6, we are likely in the sqrt-bound regime.
        
        # Let's just do:
        # If n <= p: return True
        # If n > p:
        #    If n is prime: return True (since n <= p is false, wait. If n is prime and n > p, return False)
        #    So if n is prime: return n <= p (which is False here)
        #    If n is composite: return False
        
        # So: return is_prime(n) and n <= p?
        # No, if n is composite, it's not smooth.
        # If n is prime, it's smooth iff n <= p.
        
        # So:
        if is_prime_fast(n):
            return n <= p
        else:
            return False

    # We need is_prime_fast
    def is_prime_fast(n):
        if n < 2: return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37): return True
        if n % 2 == 0 or n % 3 == 0 or n % 5 == 0 or n % 7 == 0 or n % 11 == 0 or n % 13 == 0 or n % 17 == 0 or n % 19 == 0 or n % 23 == 0 or n % 29 == 0 or n % 31 == 0 or n % 37 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    best_k = 0
    best_p = 2

    # Search
    for p in primes_list:
        if time.time() - start_time > timeout:
            break
            
        k = 0
        p_sq = p * p
        while k < 200:
            n = p_sq + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}
