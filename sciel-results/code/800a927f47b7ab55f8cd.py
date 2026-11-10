import numpy as np
import time
from math import isqrt
from fractions import Fraction
from collections import defaultdict

def _sigma_sieve(N):
    """Compute sigma(n) for all n up to N using SPF sieve."""
    spf = np.arange(N + 1, dtype=np.int64)
    sqrt_N = int(np.sqrt(N))
    for i in range(2, sqrt_N + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    rem = np.zeros(N + 1, dtype=np.int64)
    cnt = np.zeros(N + 1, dtype=np.int64)
    sigma = np.ones(N + 1, dtype=np.int64)

    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m >= 2 and spf[m] == p:
            rem[n] = rem[m]
            cnt[n] = cnt[m] + 1
        else:
            rem[n] = m
            cnt[n] = 1

        a = cnt[n]
        geo = 1
        for _ in range(a):
            geo = geo * p + 1
        sigma[n] = sigma[rem[n]] * geo

    return sigma

def _find_by_target_ratio(target_num, target_den, max_k, deadline):
    """Find k values where sigma(k)/k = target_num/target_den, i.e., k*sigma(k) = k^2 * target_num/target_den.
    Actually we need k*sigma(k) = M for fixed M. Let's think differently:
    If sigma(k)/k = r (a rational), then k*sigma(k) = k^2 * r.
    For different k with same ratio r, M differs unless k is same.
    
    Better approach: For a target ratio r = num/den, we want k such that sigma(k) = r*k.
    Then M = k*sigma(k) = r*k^2. Different k give different M.
    
    So the strategy should be: find many k with the SAME M = k*sigma(k).
    This means k1*sigma(k1) = k2*sigma(k2) = ... = M.
    
    Let's instead: sieve up to max_k, compute all M = k*sigma(k), group by M.
    But also try targeted search for specific M values.
    """
    return []

def construct():
    start = time.time()
    deadline = 110.0  # Safety margin before 150s kill

    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Strategy 1: Sieve-based scan up to various limits
    limits = [5*10**6, 10**7, 15*10**6, 2*10**7]

    for N in limits:
        if time.time() - start > deadline - 30:
            break

        try:
            sigma = _sigma_sieve(N)

            k_vals = np.arange(1, N + 1, dtype=np.int64)
            M_vals = k_vals * sigma[1:]

            mask = M_vals <= 10**18
            M_valid = M_vals[mask]
            k_valid = k_vals[mask]

            unique_M, inverse_indices, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
            max_count = int(np.max(counts))
            if max_count > best_count:
                idx = int(np.argmax(counts))
                target_M = int(unique_M[idx])
                k_solutions = k_valid[inverse_indices == idx]

                best_M = target_M
                best_k = [int(x) for x in k_solutions]
                best_count = max_count

        except MemoryError:
            break
        except Exception:
            break

    # Strategy 2: Targeted search for M values with known many preimages
    # Try to find M where k*sigma(k) = M has many solutions
    # Focus on M that are products of many small primes with appropriate exponents
    if time.time() - start < deadline - 10:
        # Search for M with structure M = 2^a * 3^b * 5^c * ... 
        # and check which k satisfy k*sigma(k) = M
        # For a given M, k must divide M, and sigma(k) = M/k
        # So k | M and sigma(k) = M/k
        # Try M values that are highly composite
        try:
            # Generate candidate M values from products of small primes
            small_primes = [2, 3, 5, 7, 11, 13]
            
            # Try M = 2^a * 3^b * 5^c for various exponents
            # and check how many k divide M with sigma(k) = M/k
            
            # Instead, let's try to find M by looking at k values that share a common ratio
            # For k = 2^a * 3^b, sigma(k) = (2^(a+1)-1)(3^(b+1)-1)/2
            # M = k*sigma(k) = 2^a * 3^b * (2^(a+1)-1) * (3^(b+1)-1) / 2
            
            # Let's just do a larger sieve if we have time
            if time.time() - start < deadline - 20:
                N2 = 3*10**7
                if N2 > 2*10**7:
                    try:
                        sigma2 = _sigma_sieve(N2)
                        k_vals2 = np.arange(1, N2 + 1, dtype=np.int64)
                        M_vals2 = k_vals2 * sigma2[1:]
                        mask2 = M_vals2 <= 10**18
                        M_valid2 = M_vals2[mask2]
                        k_valid2 = k_vals2[mask2]
                        
                        unique_M2, inverse_indices2, counts2 = np.unique(M_valid2, return_inverse=True, return_counts=True)
                        max_count2 = int(np.max(counts2))
                        if max_count2 > best_count:
                            idx2 = int(np.argmax(counts2))
                            best_M = int(unique_M2[idx2])
                            best_k = [int(x) for x in k_valid2[inverse_indices2 == idx2]]
                            best_count = max_count2
                    except MemoryError:
                        pass
                    except Exception:
                        pass
        except Exception:
            pass

    return {'M': best_M, 'k': best_k}
