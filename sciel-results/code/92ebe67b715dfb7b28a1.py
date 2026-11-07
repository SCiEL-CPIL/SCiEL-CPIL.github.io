"""Search for M with many distinct k where k*sigma(k)=M, k<=isqrt(M)."""
import math
import time
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin

    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    def linear_sigma_sieve(K):
        """Compute sigma(1..K) using linear sieve in O(K)."""
        sigma = np.zeros(K + 1, dtype=np.int64)
        sigma[0] = 0
        sigma[1] = 1
        primes = []
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        # sigma_powers[i] = p^e where p is smallest prime factor and p^e || i
        # Actually, let's use a different approach:
        # sigma(n) = sigma(rad(n)) * prod(p^(e+1)-1)/(p-1) for n = prod p^e
        # Linear sieve: maintain sigma and the highest power of smallest prime factor
        
        # Alternative: use the multiplicative property with linear sieve
        # For each n, if n = p^a * m where p doesn't divide m:
        # sigma(n) = sigma(p^a) * sigma(m)
        # We track: spf[i] = smallest prime factor, and pow_p[i] = p^a where p^a || i
        
        spf = np.zeros(K + 1, dtype=np.int64)
        pow_p = np.zeros(K + 1, dtype=np.int64)
        pow_p[1] = 1
        sigma[1] = 1
        
        for i in range(2, K + 1):
            if is_prime[i]:
                primes.append(i)
                spf[i] = i
                pow_p[i] = i
                sigma[i] = 1 + i
            for p in primes:
                if i * p > K:
                    break
                is_prime[i * p] = False
                spf[i * p] = p
                if i % p == 0:
                    # p divides i, so p^a || i means p^(a+1) || i*p
                    pow_p[i * p] = pow_p[i] * p
                    # sigma(i*p) = sigma(i) - sigma(i/p^a) * p + sigma(i/p^a) * (p^(a+1)+1)
                    # More simply: if i = p^a * m, sigma(i*p) = (1+p+...+p^(a+1)) * sigma(m)
                    # = (sigma(i) - (1+p+...+p^a-1)*sigma(m)) ... 
                    # Let's use: sigma(i*p) = sigma(i) + p * sigma(i) / (1 + p + ... + p^a) * (1 + p + ... + p^(a+1))
                    # This is getting complex. Let me use a simpler formula.
                    # If i = p^a * m, gcd(p,m)=1: sigma(i) = (p^(a+1)-1)/(p-1) * sigma(m)
                    # sigma(i*p) = (p^(a+2)-1)/(p-1) * sigma(m)
                    # So sigma(i*p) = sigma(i) * (p^(a+2)-1)/(p^(a+1)-1)
                    # = sigma(i) * (1 + 1/(p^(a+1)-1))
                    # Hmm, let me just compute directly.
                    # Actually: sigma(i*p) = sigma(i) + p * sigma(i) // ((p^(a+1)-1)//(p-1)) * ((p^(a+2)-1)//(p-1)) / ((p^(a+1)-1)//(p-1))
                    # This is messy. Let me use a different approach.
                    
                    # Simpler: maintain an array "sigma_part" where for i = p^a * m,
                    # sigma_part[i] = (1 + p + ... + p^a)
                    # Then sigma(i) = sigma_part[i] * sigma(m)
                    # But we need sigma(m) which is sigma(i) / sigma_part[i]
                    
                    # Let me just use the standard linear sieve for sigma:
                    # When i % p == 0:
                    #   pow_p[i*p] = pow_p[i] * p
                    #   sigma[i*p] = sigma[i] + pow_p[i*p] * (sigma[i] // (pow_p[i] + 1 if p==2 else (pow_p[i]*p-1)//(p-1)))
                    # This is getting too complex. Let me use a cleaner formulation.
                    
                    # Clean approach:
                    # Let n = i*p where p | i. Write i = p^a * m, p∤m.
                    # Then n = p^(a+1) * m.
                    # sigma(i) = (1+p+...+p^a) * sigma(m)
                    # sigma(n) = (1+p+...+p^(a+1)) * sigma(m)
                    # sigma(n) = sigma(i) + p^(a+1) * sigma(m)
                    # sigma(m) = sigma(i) / (1+p+...+p^a)
                    # Let S_a = 1+p+...+p^a = (p^(a+1)-1)/(p-1)
                    # sigma(n) = sigma(i) + p^(a+1) * sigma(i) / S_a
                    # = sigma(i) * (1 + p^(a+1)/S_a)
                    # = sigma(i) * (S_a + p^(a+1)) / S_a
                    # S_a + p^(a+1) = (p^(a+1)-1)/(p-1) + p^(a+1) = (p^(a+1)-1 + p^(a+1)*(p-1))/(p-1)
                    # = (p^(a+1)-1 + p^(a+2) - p^(a+1))/(p-1) = (p^(a+2)-1)/(p-1) = S_{a+1}
                    # So sigma(n) = sigma(i) * S_{a+1} / S_a
                    # And S_{a+1} = S_a + p^(a+1) = S_a + pow_p[i] * p
                    # So sigma(i*p) = sigma(i) * (S_a + pow_p[i]*p) / S_a
                    # But S_a = sigma(i) / sigma(m)... we don't have sigma(m) separately.
                    
                    # Let me just store S_a separately.
                    # Actually, the cleanest way: store for each i, the value of p^a where p^a || i (smallest prime).
                    # And store sigma. When i % p == 0:
                    #   Let p^a || i, so pow_p[i] = p^a
                    #   sigma(i) = (1+p+...+p^a) * sigma(i/p^a)
                    #   For i*p: p^(a+1) || i*p, so pow_p[i*p] = p^(a+1)
                    #   sigma(i*p) = (1+p+...+p^(a+1)) * sigma(i/p^a)
                    #   sigma(i*p) = sigma(i) * (1+p+...+p^(a+1)) / (1+p+...+p^a)
                    #   = sigma(i) * (1 + p^(a+1)/(1+p+...+p^a))
                    #   Let S = 1+p+...+p^a. Then sigma(i*p) = sigma(i) * (S + p^(a+1)) / S
                    #   But S = (p^(a+1)-1)/(p-1), and p^(a+1) = pow_p[i] * p
                    #   S + p^(a+1) = (p^(a+1)-1)/(p-1) + p^(a+1) = (p^(a+1)-1 + p^(a+2)-p^(a+1))/(p-1) = (p^(a+2)-1)/(p-1)
                    #   So ratio = (p^(a+2)-1)/(p^(a+1)-1)
                    #   sigma(i*p) = sigma(i) * (p^(a+2)-1) / (p^(a+1)-1)
                    #   Let A = p^(a+1) = pow_p[i] * p
                    #   Then p^(a+2) = A*p
                    #   ratio = (A*p - 1) / (A - 1)
                    #   sigma(i*p) = sigma(i) * (A*p - 1) / (A - 1)
                    A = pow_p[i] * p  # This is p^(a+1)
                    # sigma(i*p) = sigma(i) * (A*p - 1) / (A - 1)
                    # This should be an integer.
                    sigma[i * p] = sigma[i] * (A * p - 1) // (A - 1)
                    break
                else:
                    # p does not divide i, so p is a new prime factor
                    pow_p[i * p] = p
                    sigma[i * p] = sigma[i] * (1 + p)
        
        return sigma

    def find_best_from_K(K, deadline, best_count, best_M, best_k):
        """Given K, compute sigma, group, return updated best."""
        if time.time() > deadline - 20:
            return best_M, best_k, best_count

        sigma = linear_sigma_sieve(K)

        if time.time() > deadline - 10:
            return best_M, best_k, best_count

        ks_range = np.arange(1, K + 1, dtype=np.int64)
        Ms = ks_range * sigma[1:]

        # Only consider M <= 10^18
        valid_mask = Ms <= 10**18
        Ms_valid = Ms[valid_mask]
        ks_valid = ks_range[valid_mask]

        if len(Ms_valid) == 0:
            return best_M, best_k, best_count

        # Group by M
        unique_Ms, inverse, counts = np.unique(Ms_valid, return_inverse=True, return_counts=True)

        max_count_idx = np.argmax(counts)
        max_count = int(counts[max_count_idx])

        if max_count > best_count:
            best_M = int(unique_Ms[max_count_idx])
            # Find all k with this M
            k_indices = np.where(Ms_valid == best_M)[0]
            best_k = [int(x) for x in ks_valid[k_indices]]
            best_count = max_count

        return best_M, best_k, best_count

    try:
        # Main search: try larger K with faster sieve
        for K_try in [20000000, 15000000, 10000000]:
            if time.time() > deadline - 30:
                break
            try:
                best_M, best_k, best_count = find_best_from_K(
                    K_try, deadline, best_count, best_M, best_k
                )
                break
            except MemoryError:
                continue
    except Exception:
        pass

    # Secondary: smooth-number focused search for higher density
    if time.time() < deadline - 25:
        try:
            small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            max_K_smooth = 5000000
            max_exponents = []
            for p in small_primes:
                e = 0
                val = 1
                while val * p <= max_K_smooth:
                    val *= p
                    e += 1
                max_exponents.append(e)

            smooth_nums = []

            def gen_smooth(idx, current):
                if time.time() > deadline - 15:
                    return
                if current > max_K_smooth:
                    return
                if idx == len(small_primes):
                    if current > 1:
                        smooth_nums.append(current)
                    return
                p = small_primes[idx]
                max_e = max_exponents[idx]
                for e in range(max_e + 1):
                    if current * (p ** e) > max_K_smooth:
                        break
                    gen_smooth(idx + 1, current * (p ** e))

            gen_smooth(0, 1)

            if len(smooth_nums) > 0:
                sigma_dict = {}
                for k in smooth_nums:
                    if k in sigma_dict:
                        continue
                    n = k
                    sigma_val = 1
                    for p in small_primes:
                        if n % p == 0:
                            pe = 1
                            p_term = 1
                            while n % p == 0:
                                n //= p
                                pe *= p
                                p_term += pe
                            sigma_val *= p_term
                    if n > 1:
                        sigma_val *= (1 + n)
                    sigma_dict[k] = sigma_val

                groups = defaultdict(list)
                for k in smooth_nums:
                    M = k * sigma_dict[k]
                    if M <= 10**18:
                        groups[M].append(k)

                for M, ks in groups.items():
                    cnt = len(ks)
                    if cnt > best_count:
                        best_count = cnt
                        best_M = M
                        best_k = ks

        except Exception:
            pass

    # Ensure types are plain Python int and list
    best_k = [int(x) for x in best_k]
    best_M = int(best_M)

    return {'M': best_M, 'k': best_k}
