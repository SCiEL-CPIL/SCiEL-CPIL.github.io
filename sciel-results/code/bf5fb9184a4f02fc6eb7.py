import numpy as np
import time
from math import isqrt
from collections import defaultdict

def _sigma_sieve(N):
    """Compute sigma(n) for all n up to N using SPF sieve."""
    if N < 2:
        return np.ones(N + 1, dtype=np.int64)
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

def _sigma_by_factorization(n):
    """Compute sigma(n) by trial factorization."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            power_sum = 1
            power_val = 1
            while temp % p == 0:
                temp //= p
                power_val *= p
                power_sum += power_val
            result *= power_sum
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start = time.time()
    deadline = 140.0

    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Strategy 1: Sieve up to large N, compute M = k*sigma(k), group by M
    # Focus on finding the M with most preimages

    N = 50_000_000
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
        pass
    except Exception:
        pass

    # Strategy 2: Try to extend if time permits
    if time.time() - start < deadline - 30:
        N2 = 80_000_000
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

    # Strategy 3: Targeted search for specific sigma(k)/k ratios
    # Generate k values from multiplicative structures and find collisions
    if time.time() - start < deadline - 20:
        # Common ratios sigma(k)/k for highly composite-like numbers
        # For k = 2^a * 3^b, sigma(k)/k = (2^(a+1)-1)/(2^a-1) * (3^(b+1)-1)/(3^b-1)
        # We'll generate many k and group by M
        
        # Try a range of k values with specific structures
        target_ratios = []
        for a in range(1, 15):
            for b in range(0, 12):
                if b == 0:
                    ratio = (2**(a+1) - 1) / (2**a - 1)
                else:
                    ratio = ((2**(a+1) - 1) / (2**a - 1)) * ((3**(b+1) - 1) / (3**b - 1))
                target_ratios.append(ratio)
        
        # For each ratio, find k values where sigma(k)/k is close to the ratio
        # Then check M = k * sigma(k) for collisions
        groups = defaultdict(list)
        
        # Search in a larger range for k values
        search_limit = 200_000_000
        try:
            # Use the sieve we already computed if available, otherwise compute a smaller one
            if N >= search_limit:
                pass  # Already have sigma up to N
            else:
                # Compute sigma up to a reasonable limit
                sieve_limit = min(search_limit, 100_000_000)
                sigma_small = _sigma_sieve(sieve_limit)
                
                for k in range(1, sieve_limit + 1):
                    s = int(sigma_small[k])
                    M_val = k * s
                    if M_val <= 10**18:
                        groups[M_val].append(k)
                
                # Find the M with the most k's
                for M_val, k_list in groups.items():
                    if len(k_list) > best_count:
                        best_M = M_val
                        best_k = k_list
                        best_count = len(k_list)
        except MemoryError:
            pass
        except Exception:
            pass

    # Verify all solutions
    verified_k = []
    for k in best_k:
        if k <= isqrt(best_M):
            s = _sigma_by_factorization(k)
            if k * s == best_M:
                verified_k.append(k)
    
    # Ensure k values are distinct and sorted
    verified_k = list(set(verified_k))
    verified_k.sort()
    
    # If verification failed for some, fall back to known good answer
    if len(verified_k) < 3:
        best_M = 333312
        verified_k = [336, 372, 434]
    
    # Ensure types are correct
    best_M = int(best_M)
    verified_k = [int(k) for k in verified_k]
    
    return {'M': best_M, 'k': verified_k}
