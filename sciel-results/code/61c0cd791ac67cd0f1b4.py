"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute small primes for trial division
    primes = _sieve_primes(10000)
    n_primes = len(primes)
    
    best_val = float('inf')
    best_n = 10000
    
    def largest_prime_factor(x, current_best):
        """Return largest prime factor of x, or early-exit if >= current_best."""
        if x <= 1:
            return 1
        lpf = 1
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                lpf = p
                while x % p == 0:
                    x //= p
                if lpf >= current_best:
                    return lpf
        if x > 1:
            lpf = max(lpf, x)
        return lpf
    
    # Search the range
    for n in range(10000, 1000001):
        if time.time() > deadline:
            break
        
        # Check n
        lpf_n = largest_prime_factor(n, best_val)
        if lpf_n >= best_val:
            continue
        
        # Check n+1
        lpf_np1 = largest_prime_factor(n + 1, best_val)
        if lpf_np1 >= best_val:
            continue
        
        candidate = max(lpf_n, lpf_np1)
        if candidate < best_val:
            best_val = candidate
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
