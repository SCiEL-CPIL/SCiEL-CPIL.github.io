"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def _sigma_linear_sieve(n, deadline=None):
    """Compute sigma(k) for k=1..n using a linear sieve. O(n) time."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    sigma[0] = 0
    if n >= 1:
        sigma[1] = 1
    is_prime = np.ones(n + 1, dtype=bool)
    primes = []
    # smallest_prime_factor and prime_power_sigma for linear sieve
    spf = np.zeros(n + 1, dtype=np.int64)  # smallest prime factor
    pp_sigma = np.zeros(n + 1, dtype=np.int64)  # sigma of p^e where p^e is the highest power of spf[i] dividing i
    
    for i in range(2, n + 1):
        if deadline is not None and time.time() > deadline:
            break
        if is_prime[i]:
            primes.append(i)
            spf[i] = i
            pp_sigma[i] = 1 + i  # sigma(p) = 1 + p
            sigma[i] = 1 + i
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            spf[i * p] = p
            if p == spf[i]:
                # p divides i, so p^e * i' where i' not divisible by p
                # pp_sigma[i*p] = pp_sigma[i] * p + p^(e+1) ... actually:
                # if i = p^e * m, gcd(m,p)=1, then i*p = p^(e+1)*m
                # pp_sigma[i*p] = sigma(p^(e+1)) = pp_sigma[i] * p + p^(e+1)
                # But we need to track p^(e+1). Let's use a different approach.
                # pp_sigma[i] = sigma(p^e) where p^e || i
                # For i*p: p^(e+1) || i*p, so pp_sigma[i*p] = sigma(p^(e+1))
                # sigma(p^(e+1)) = (p^(e+2)-1)/(p-1) = p * (p^(e+1)-1)/(p-1) + p^(e+1)
                # = p * (pp_sigma[i] - p^e) + p^(e+1)
                # This is getting complex. Let me use a simpler approach.
                
                # Actually, let me just use the standard formula:
                # sigma(i*p) = sigma(i) * (1 + p) if gcd(i,p)=1
                # sigma(i*p) = sigma(i) * (p^(e+2)-1)/(p-1) / ((p^(e+1)-1)/(p-1)) * ... 
                # = sigma(i) * (1 + p + ... + p^(e+1)) / (1 + p + ... + p^e)
                
                # Simpler: track for each i, the prime power part
                # Let me just compute it directly for the linear sieve
                # sigma(i*p) when p|i: let i = p^e * m, gcd(m,p)=1
                # sigma(i) = sigma(p^e) * sigma(m)
                # sigma(i*p) = sigma(p^(e+1)) * sigma(m)
                # So sigma(i*p) = sigma(i) * sigma(p^(e+1)) / sigma(p^e)
                # = sigma(i) * (1 + p^(e+1)) / (1 + p^e) ... no that's wrong
                # sigma(p^e) = (p^(e+1)-1)/(p-1)
                # sigma(p^(e+1)) = (p^(e+2)-1)/(p-1)
                # ratio = (p^(e+2)-1)/(p^(e+1)-1)
                
                # This is getting too complex for numpy. Let me use a different strategy.
                # I'll compute sigma using the multiplicative property with a modified sieve.
                break
            else:
                # gcd(i,p) = 1, so sigma(i*p) = sigma(i) * sigma(p) = sigma(i) * (1+p)
                spf[i * p] = p
                pp_sigma[i * p] = 1 + p
                sigma[i * p] = sigma[i] * (1 + p)
    
    # The above linear sieve has issues with the p|i case. Let me use a simpler O(n log n) 
    # but more efficient approach, or fix the linear sieve.
    
    # Let me restart with a cleaner approach: use the standard divisor sum sieve
    # but with better numpy optimization
    return sigma

def _sigma_sieve_fast(n, deadline=None):
    """Compute sigma(k) for k=1..n efficiently."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        if deadline is not None and time.time() > deadline:
            break
        # Add i to all multiples of i
        for j in range(i, n + 1, i):
            sigma[j] += i
    return sigma

def _sigma_sieve_numpy(n, deadline=None):
    """Compute sigma(k) for k=1..n using numpy slicing (fast for moderate n)."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        if deadline is not None and time.time() > deadline:
            break
        sigma[i::i] += i
    return sigma

def _sigma_linear_sieve_correct(n, deadline=None):
    """Compute sigma(k) for k=1..n using a correct linear sieve."""
    if n < 1:
        return np.zeros(1, dtype=np.int64)
    
    sigma = np.zeros(n + 1, dtype=np.int64)
    sigma[1] = 1
    
    is_composite = np.zeros(n + 1, dtype=bool)
    primes = []
    # For the linear sieve, we need to track:
    # spf[i] = smallest prime factor of i
    # pp_sigma[i] = sigma(p^e) where p^e is the exact power of spf[i] in i
    # pp_val[i] = p^e (the actual prime power)
    
    spf = np.zeros(n + 1, dtype=np.int64)
    pp_sigma = np.zeros(n + 1, dtype=np.int64)
    pp_val = np.zeros(n + 1, dtype=np.int64)
    
    for i in range(2, n + 1):
        if deadline is not None and time.time() > deadline:
            break
        if not is_composite[i]:
            primes.append(i)
            spf[i] = i
            pp_sigma[i] = 1 + i  # sigma(p) = 1+p
            pp_val[i] = i
            sigma[i] = 1 + i
        for p in primes:
            if i * p > n:
                break
            is_composite[i * p] = True
            spf[i * p] = p
            if p == spf[i]:
                # p divides i, so i = p^e * m with gcd(m,p)=1
                # i*p = p^(e+1) * m
                # pp_val[i*p] = pp_val[i] * p
                # pp_sigma[i*p] = pp_sigma[i] + pp_val[i] * p  (since sigma(p^(e+1)) = sigma(p^e) + p^(e+1))
                pp_val[i * p] = pp_val[i] * p
                pp_sigma[i * p] = pp_sigma[i] + pp_val[i * p]
                # sigma(i*p) = sigma(m) * pp_sigma[i*p]
                # sigma(i) = sigma(m) * pp_sigma[i]
                # So sigma(i*p) = sigma(i) * pp_sigma[i*p] // pp_sigma[i]
                # But integer division might lose precision. Let's compute differently.
                # Actually: sigma(i*p) = (sigma(i) // pp_sigma[i]) * pp_sigma[i*p]
                # This works if pp_sigma[i] divides sigma(i), which it does since sigma is multiplicative.
                sigma[i * p] = (sigma[i] // pp_sigma[i]) * pp_sigma[i * p]
                break
            else:
                # gcd(i, p) = 1
                spf[i * p] = p
                pp_val[i * p] = p
                pp_sigma[i * p] = 1 + p
                sigma[i * p] = sigma[i] * (1 + p)
    
    return sigma

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with known good result
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    def _sigma_verify(k):
        """Compute sigma(k) by trial division."""
        if k == 1:
            return 1
        result = 1
        temp = k
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                result *= (p**(exp+1) - 1) // (p - 1)
            p += 1 if p == 2 else 2
        if temp > 1:
            result *= (temp**2 - 1) // (temp - 1)
        return result
    
    # Phase 1: Use linear sieve for larger n
    # Try increasing sizes with the faster linear sieve
    sieve_sizes = [2000000, 5000000, 10000000, 20000000, 50000000]
    
    for n in sieve_sizes:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        # Estimate: linear sieve is O(n), should be much faster
        # For n=50M, should take maybe 5-15 seconds
        if n > 10000000 and elapsed > 40:
            break
        if n > 20000000 and elapsed > 70:
            break
        if n > 50000000 and elapsed > 100:
            break
        
        sigma = _sigma_linear_sieve_correct(n, deadline=deadline - 20)
        
        if time.time() - start_time > deadline - 15:
            break
        
        # Group k by M = k * sigma(k)
        groups = defaultdict(list)
        
        ks_array = np.arange(1, n + 1, dtype=np.int64)
        M_values = ks_array * sigma[1:]
        
        # Filter to M <= 10^18
        mask = M_values <= 10**18
        valid_M = M_values[mask]
        valid_k = ks_array[mask]
        
        # Process in chunks
        chunk_size = 500000
        for start in range(0, len(valid_M), chunk_size):
            end = min(start + chunk_size, len(valid_M))
            for i in range(start, end):
                m = int(valid_M[i])
                k = int(valid_k[i])
                groups[m].append(k)
            
            if time.time() - start_time > deadline - 10:
                break
        
        # Find M with most solutions
        for M, ks in groups.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_ks = ks[:1000]
        
        del groups
        del sigma
        del M_values
        del valid_M
        del valid_k
    
    # Phase 2: Targeted search for high-collision M values
    # Strategy: look for M where many k have k*sigma(k) = M
    # This happens when k has specific multiplicative structures
    # Try specific families of k
    
    if time.time() - start_time < deadline - 20:
        # Try to find more solutions by searching around known good M values
        # and by exploring structured k values
        
        # Check if we can find more solutions for the current best_M
        if best_count >= 3:
            # Try to find additional k values that satisfy k*sigma(k) = best_M
            # by searching k up to isqrt(best_M)
            max_k = math.isqrt(best_M)
            # This could be very large, so only do it if max_k is reasonable
            if max_k <= 10000000 and time.time() - start_time < deadline - 15:
                # Use the sieve if we have one, or compute sigma on the fly
                # For now, just verify what we have
                pass
    
    # Verify the result
    verified_ks = []
    for k in best_ks:
        if k * _sigma_verify(k) == best_M and k <= math.isqrt(best_M):
            verified_ks.append(k)
    
    if len(verified_ks) >= 1:
        return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}
# EVOLVE-BLOCK-END
