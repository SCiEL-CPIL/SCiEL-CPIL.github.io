import numpy as np
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Fallback: known M with 3 solutions
    fallback_M = 333312
    fallback_k = [336, 372, 434]
    
    best_M = fallback_M
    best_ks = list(fallback_k)
    best_count = len(fallback_k)
    
    # Phase 1: NumPy sieve for sigma, group by M
    # Try increasing N until time runs out
    N_attempts = [10_000_000, 20_000_000, 40_000_000, 80_000_000]
    
    M_to_ks = defaultdict(list)
    
    for N in N_attempts:
        if time.time() > deadline - 15:
            break
        
        try:
            # Compute sigma(1..N) using NumPy sieve
            sigma = np.zeros(N + 1, dtype=np.int64)
            
            for i in range(1, N + 1):
                sigma[i::i] += i
                if i % 5_000_000 == 0 and time.time() > deadline - 10:
                    raise TimeoutError()
            
            k_vals = np.arange(1, N + 1, dtype=np.int64)
            M_vals = k_vals * sigma[1:]
            
            # Filter M <= 10^18
            mask = M_vals <= 10**18
            k_valid = k_vals[mask]
            M_valid = M_vals[mask]
            
            # Group by M using np.unique
            unique_M, inverse, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
            
            for idx in range(len(unique_M)):
                if counts[idx] > best_count:
                    best_idx = idx
                    best_M = int(unique_M[best_idx])
                    k_indices = np.where(inverse == best_idx)[0]
                    best_ks = [int(k) for k in k_valid[k_indices]]
                    best_count = int(counts[best_idx])
            
            # Also store in M_to_ks for potential Phase 2 use
            # (only for M with count >= 3 to save memory)
            for idx in range(len(unique_M)):
                if counts[idx] >= 3:
                    M_val = int(unique_M[idx])
                    k_indices = np.where(inverse == idx)[0]
                    M_to_ks[M_val] = [int(k) for k in k_valid[k_indices]]
                    
        except MemoryError:
            continue
        except TimeoutError:
            break
        except Exception:
            continue
    
    # Phase 2: Targeted search for smooth M
    # Generate M values that are products of small primes (smooth numbers)
    # For each smooth M, find all k such that k*sigma(k) = M
    # This is expensive, so limit to promising M ranges
    
    if time.time() < deadline - 20:
        # Generate candidate M values: products of first few primes with small exponents
        # M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * ...
        # We want M <= 10^18
        
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        
        def gen_smooth(primes, max_val, max_depth=0):
            """Generate smooth numbers up to max_val"""
            if max_depth == len(primes):
                yield 1
                return
            p = primes[max_depth]
            power = 1
            while power <= max_val:
                for rest in gen_smooth(primes, max_val // power, max_depth + 1):
                    yield power * rest
                power *= p
        
        # Generate smooth M candidates
        smooth_Ms = set()
        for m in gen_smooth(primes, 10**18):
            smooth_Ms.add(m)
        
        # For each smooth M, find k such that k*sigma(k) = M
        # k must divide M, so enumerate divisors of M
        # But M can be huge, so limit to M where we can enumerate divisors
        
        # Better approach: for each k up to some limit, check if k*sigma(k) is smooth
        # We already did this in Phase 1. Let's instead focus on M that appeared
        # multiple times in Phase 1 and verify/extend.
        
        # Alternative: direct search for k in ranges where sigma(k)/k is "nice"
        # k = 2^a * 3^b * 5^c ... (smooth k)
        # sigma(k) = (2^(a+1)-1) * (3^(b+1)-1)/2 * ...
        # M = k * sigma(k)
        
        # Enumerate smooth k up to some limit
        smooth_k_limit = 500000
        smooth_k_to_M = defaultdict(list)
        
        def gen_smooth_k(primes, max_val, current, max_depth=0):
            """Generate smooth numbers up to max_val"""
            if max_depth == len(primes):
                if current > 0 and current <= max_val:
                    yield current
                return
            p = primes[max_depth]
            power = 1
            while current * power <= max_val:
                yield from gen_smooth_k(primes, max_val, current * power, max_depth + 1)
                power *= p
        
        # Use a different approach: iterative generation
        smooth_ks = [1]
        for p in primes:
            new_smooth = []
            for sk in smooth_ks:
                val = sk
                while val <= smooth_k_limit:
                    new_smooth.append(val)
                    val *= p
            smooth_ks = new_smooth
        smooth_ks = sorted(set(smooth_ks))
        
        # For each smooth k, compute M = k * sigma(k) and group
        for k in smooth_ks:
            if time.time() > deadline - 5:
                break
            M = k * _sigma_fast(k)
            if M <= 10**18:
                smooth_k_to_M[M].append(k)
        
        # Find best M from smooth k search
        for M, ks in smooth_k_to_M.items():
            if len(ks) > best_count:
                # Verify these k values
                verified = [k for k in ks if k * _sigma_fast(k) == M and k <= math.isqrt(M)]
                if len(verified) > best_count:
                    best_M = M
                    best_ks = verified
                    best_count = len(verified)
    
    # Final verification
    verified_ks = []
    for k in best_ks:
        try:
            if k * _sigma_fast(k) == best_M and k <= math.isqrt(best_M) and k >= 1:
                verified_ks.append(int(k))
        except:
            pass
    
    # Deduplicate
    verified_ks = list(set(verified_ks))
    verified_ks.sort()
    
    if len(verified_ks) > best_count:
        best_count = len(verified_ks)
        best_ks = verified_ks
    
    if not best_ks:
        best_M = fallback_M
        best_ks = fallback_k
        best_count = len(fallback_k)
    
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}


def _sigma_fast(n):
    """Faster sigma using prime trial division with 6k±1 optimization."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    result = 1
    temp = n
    # Factor out 2
    if temp % 2 == 0:
        p_power = 1
        p_sum = 1
        while temp % 2 == 0:
            temp //= 2
            p_power *= 2
            p_sum += p_power
        result *= p_sum
    # Factor out 3
    if temp % 3 == 0:
        p_power = 1
        p_sum = 1
        while temp % 3 == 0:
            temp //= 3
            p_power *= 3
            p_sum += p_power
        result *= p_sum
    # Try 6k±1
    d = 5
    step = 2
    while d * d <= temp:
        if temp % d == 0:
            p_power = 1
            p_sum = 1
            while temp % d == 0:
                temp //= d
                p_power *= d
                p_sum += p_power
            result *= p_sum
        d += step
        step = 6 - step
    if temp > 1:
        result *= (1 + temp)
    return result
