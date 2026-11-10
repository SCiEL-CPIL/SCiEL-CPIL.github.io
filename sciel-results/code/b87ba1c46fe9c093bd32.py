"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

MAX_VAL = 10_000_000
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _build_spf(n):
    """Build smallest prime factor array up to n."""
    spf = list(range(n + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(n**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _sigma(n, spf):
    """Compute sigma(n) using SPF."""
    if n == 1:
        return 1
    result = 1
    while n > 1:
        p = spf[n]
        # Collect power of p
        power = 1
        while n % p == 0:
            n //= p
            power *= p
        # sigma(p^k) = (p^(k+1) - 1)/(p - 1)
        result *= (power * p - 1) // (p - 1)
    return result

def construct():
    start_time = time.time()
    
    # Build SPF table
    spf = _build_spf(MAX_VAL)
    
    best_error = None
    best_pair = (11, 6)  # fallback: sigma(11)=12, sigma(6)=12, |11/6 - 3/2| = |1.8333 - 1.5| = 0.3333
    
    # Search over b values
    # a ≈ 3b/2, need a <= MAX_VAL so b <= 2*MAX_VAL/3
    max_b = (2 * MAX_VAL) // 3
    
    # We'll search b from 1 to max_b
    # For efficiency, we can search in chunks and check time periodically
    for b in range(1, max_b + 1):
        # Check time every 10000 iterations
        if b % 10000 == 0:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        # Candidate a values: floor(3b/2) and ceil(3b/2)
        # 3b/2 might not be integer
        if b % 2 == 0:
            a_exact = 3 * b // 2
            candidates = [a_exact]
        else:
            a_floor = (3 * b) // 2
            a_ceil = a_floor + 1
            candidates = [a_floor, a_ceil]
        
        sigma_b = _sigma(b, spf)
        
        for a in candidates:
            if a < 1 or a > MAX_VAL:
                continue
            # Skip if a/b == 3/2 exactly (error = 0, invalid)
            if 2 * a == 3 * b:
                continue
            
            sigma_a = _sigma(a, spf)
            if sigma_a == sigma_b:
                # Compute error |a/b - 3/2| = |2a - 3b| / (2b)
                error_num = abs(2 * a - 3 * b)
                error_den = 2 * b
                # error = error_num / error_den
                if best_error is None:
                    best_error = error_num / error_den
                    best_pair = (a, b)
                else:
                    # Compare error_num/error_den vs best_error
                    # best_error is stored as float, but let's compare exactly
                    # error_num/error_den < best_error_num/best_error_den
                    # We need to store best as fraction
                    pass
        
        # Actually, let me redo this with exact fraction comparison
    # Let me restructure to track best as (num, den) fraction
    
    # Restart with proper tracking
    start_time = time.time()
    best_num = None
    best_den = None
    best_pair = (11, 6)
    
    max_b = (2 * MAX_VAL) // 3
    
    for b in range(1, max_b + 1):
        if b % 20000 == 0:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
        
        if b % 2 == 0:
            a_exact = 3 * b // 2
            candidates = [a_exact]
        else:
            a_floor = (3 * b) // 2
            a_ceil = a_floor + 1
            candidates = [a_floor, a_ceil]
        
        sigma_b = _sigma(b, spf)
        
        for a in candidates:
            if a < 1 or a > MAX_VAL:
                continue
            if 2 * a == 3 * b:
                continue
            
            sigma_a = _sigma(a, spf)
            if sigma_a == sigma_b:
                error_num = abs(2 * a - 3 * b)
                error_den = 2 * b
                
                if best_num is None:
                    best_num = error_num
                    best_den = error_den
                    best_pair = (a, b)
                else:
                    # Compare error_num/error_den < best_num/best_den
                    # error_num * best_den < best_num * error_den
                    if error_num * best_den < best_num * error_den:
                        best_num = error_num
                        best_den = error_den
                        best_pair = (a, b)
    
    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
