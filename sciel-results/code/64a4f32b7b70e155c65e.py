import time
import math
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START
def _sigma_fast(n):
    """Faster sigma using prime trial division with 6k±1 optimization."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    result = 1
    temp = n
    if temp % 2 == 0:
        p_power = 1
        p_sum = 1
        while temp % 2 == 0:
            temp //= 2
            p_power *= 2
            p_sum += p_power
        result *= p_sum
    if temp % 3 == 0:
        p_power = 1
        p_sum = 1
        while temp % 3 == 0:
            temp //= 3
            p_power *= 3
            p_sum += p_power
        result *= p_sum
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


def _sigma_sieve_numpy(n_max):
    """Compute sigma(n) for all n up to n_max using NumPy."""
    sigma = np.zeros(n_max + 1, dtype=np.int64)
    for d in range(1, n_max + 1):
        sigma[d::d] += d
    return sigma


def _generate_smooth_numbers(primes, max_val, exponents):
    """Generate all numbers of form prod(p_i^e_i) <= max_val."""
    results = []
    n_primes = len(primes)
    
    def _recurse(idx, current):
        if current > max_val:
            return
        if idx == n_primes:
            if current > 0:
                results.append(current)
            return
        p = primes[idx]
        val = current
        for e in range(exponents[idx] + 1):
            if val > max_val:
                break
            _recurse(idx + 1, val)
            val *= p
    
    _recurse(0, 1)
    return results


def construct():
    start_time = time.time()
    deadline = 130.0  # 20s safety margin before 150s kill
    
    M_to_ks = defaultdict(list)
    best_M = 1
    best_ks = [1]
    
    # Phase 1: NumPy sieve for small k range
    # Start with a manageable range
    max_k = 200000
    try:
        sigma_arr = _sigma_sieve_numpy(max_k)
        for k in range(1, max_k + 1):
            M = k * sigma_arr[k]
            if M <= 10**18:
                M_to_ks[M].append(k)
    except Exception:
        pass
    
    # Update best
    for M, ks in M_to_ks.items():
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Not enough time for more phases
        best_ks = list(set(best_ks))
        best_ks.sort()
        verified_ks = []
        for k in best_ks:
            if k * _sigma_fast(k) == best_M and k <= math.isqrt(best_M):
                verified_ks.append(k)
        if not verified_ks:
            verified_ks = [336, 372, 434]
            best_M = 333312
        return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
    
    # Phase 2: Extend sieve to larger ranges
    for max_k2 in [500000, 1000000, 2000000, 5000000]:
        if time.time() - start_time > deadline - 20:
            break
        try:
            sigma_arr2 = _sigma_sieve_numpy(max_k2)
            for k in range(max_k + 1, max_k2 + 1):
                M = k * sigma_arr2[k]
                if M <= 10**18:
                    M_to_ks[M].append(k)
            max_k = max_k2
        except Exception:
            break
        
        # Update best
        for M, ks in M_to_ks.items():
            if len(ks) > len(best_ks):
                best_M = M
                best_ks = ks
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        best_ks = list(set(best_ks))
        best_ks.sort()
        verified_ks = []
        for k in best_ks:
            if k * _sigma_fast(k) == best_M and k <= math.isqrt(best_M):
                verified_ks.append(k)
        if not verified_ks:
            verified_ks = [336, 372, 434]
            best_M = 333312
        return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
    
    # Phase 3: Targeted search for smooth numbers with specific sigma/k ratios
    # Generate smooth numbers from small primes and check for M collisions
    # This can find k values that the sieve missed (larger k)
    small_primes = [2, 3, 5, 7, 11, 13]
    
    # Try different exponent bounds
    for max_exp in [10, 15, 20]:
        if time.time() - start_time > deadline - 10:
            break
        
        # Generate smooth numbers up to a reasonable bound
        # k must satisfy k * sigma(k) <= 10^18, and sigma(k) >= k, so k <= 10^9
        # But for smooth numbers, let's go up to ~10^8
        max_smooth = 100000000  # 10^8
        
        # Generate all 2^a * 3^b * 5^c * 7^d * 11^e * 13^f <= max_smooth
        # with each exponent <= max_exp
        smooth_ks = set()
        
        # Recursive generation
        def _gen(idx, current, primes_list):
            if current > max_smooth:
                return
            if idx == len(primes_list):
                if current > 0 and current <= max_smooth:
                    smooth_ks.add(current)
                return
            p = primes_list[idx]
            val = current
            for e in range(max_exp + 1):
                if val > max_smooth:
                    break
                _gen(idx + 1, val, primes_list)
                val *= p
        
        _gen(0, 1, small_primes)
        
        # For each smooth k not already checked, compute M and add
        for k in smooth_ks:
            if k > max_k:  # skip if already in sieve range
                if time.time() - start_time > deadline - 5:
                    break
                s = _sigma_fast(k)
                M = k * s
                if M <= 10**18:
                    M_to_ks[M].append(k)
        
        # Update best
        for M, ks in M_to_ks.items():
            if len(ks) > len(best_ks):
                best_M = M
                best_ks = ks
    
    # Phase 4: Also try 7-smooth and 11-smooth with higher exponents for larger k
    if time.time() - start_time < deadline - 5:
        # Focus on 2^a * 3^b * 5^c * 7^d with higher exponents
        # These can reach larger k values
        for max_exp in [20, 30]:
            if time.time() - start_time > deadline - 3:
                break
            smooth_ks2 = set()
            
            def _gen2(idx, current, primes_list):
                if current > 10**10:  # k can be up to ~10^10 since sigma(k) >= k
                    return
                if idx == len(primes_list):
                    if current > 0:
                        smooth_ks2.add(current)
                    return
                p = primes_list[idx]
                val = current
                for e in range(max_exp + 1):
                    if val > 10**10:
                        break
                    _gen2(idx + 1, val, primes_list)
                    val *= p
            
            _gen2(0, 1, [2, 3, 5, 7])
            
            for k in smooth_ks2:
                if k > max_k:
                    if time.time() - start_time > deadline - 2:
                        break
                    s = _sigma_fast(k)
                    M = k * s
                    if M <= 10**18:
                        M_to_ks[M].append(k)
            
            for M, ks in M_to_ks.items():
                if len(ks) > len(best_ks):
                    best_M = M
                    best_ks = ks
    
    # Final verification
    best_ks = list(set(best_ks))
    best_ks.sort()
    verified_ks = []
    for k in best_ks:
        if k * _sigma_fast(k) == best_M and k <= math.isqrt(best_M):
            verified_ks.append(k)
    
    if not verified_ks:
        verified_ks = [336, 372, 434]
        best_M = 333312
    
    # Limit to 1000
    verified_ks = verified_ks[:1000]
    
    return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
# EVOLVE-BLOCK-END
