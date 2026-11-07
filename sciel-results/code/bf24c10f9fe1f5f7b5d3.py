import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    # Initialize with known seed
    best_M = 333312
    best_ks = [336, 372, 434]
    
    def update_best(M, ks):
        nonlocal best_M, best_ks
        # Convert numpy arrays to python lists/ints for return
        if isinstance(ks, np.ndarray):
            ks_list = [int(x) for x in ks]
        else:
            ks_list = list(ks)
            
        if len(ks_list) > len(best_ks):
            best_M = int(M)
            best_ks = ks_list
    
    def sigma_func(n):
        """Compute sigma(n) - sum of divisors of n."""
        if n <= 1:
            return n
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                power_sum = 1
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    power_sum += power
                result *= power_sum
            p += 1
        if temp > 1:
            result *= (1 + temp)
        return result
    
    def is_valid_solution(k, M):
        """Check if k*sigma(k) == M exactly."""
        if k <= 0 or k > M:
            return False
        sig = sigma_func(k)
        return k * sig == M
    
    def search_targeted_M():
        """
        Search for M with many k solutions by constructing M from prime powers.
        For each prime p, we can choose exponent a_p. Then k = prod p^b_p where 0 <= b_p <= a_p.
        k*sigma(k) = prod p^(2*b_p+1) * (p^(b_p+1)-1)/(p-1)
        We want to maximize the number of distinct k that give the same M.
        """
        local_start = time.time()
        
        # Small primes to consider
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        max_M = 10**18
        
        # For a given set of primes with max exponents, find all k that satisfy k*sigma(k)=M
        # Strategy: pick a target M structure, then enumerate valid k
        
        # Key insight: if M = prod p_i^(e_i), and k = prod p_i^(f_i) where f_i <= e_i,
        # then k*sigma(k) = prod [p_i^(2*f_i+1) * (p_i^(f_i+1)-1)/(p_i-1)]
        # For this to equal M, we need the product to match exactly.
        
        # Simpler approach: generate candidate M values with many small prime factors,
        # then for each M, try to find k values.
        
        # Generate M as products of small prime powers
        # M = 2^a * 3^b * 5^c * ... 
        # Try various combinations
        
        candidates = []
        
        # Generate M values: product of first n primes, each to some power
        # Start with highly composite-like numbers
        
        # Try M = 2^a * 3^b * 5^c * 7^d * 11^e for small exponents
        for a in range(1, 20):
            if time.time() > deadline - 30:
                break
            M_2 = 2**a
            if M_2 > max_M:
                break
            for b in range(0, 15):
                if time.time() > deadline - 30:
                    break
                M_3 = M_2 * (3**b)
                if M_3 > max_M:
                    break
                for c in range(0, 10):
                    if time.time() > deadline - 30:
                        break
                    M_5 = M_3 * (5**c)
                    if M_5 > max_M:
                        break
                    for d in range(0, 8):
                        if time.time() > deadline - 30:
                            break
                        M_7 = M_5 * (7**d)
                        if M_7 > max_M:
                            break
                        for e in range(0, 6):
                            if time.time() > deadline - 30:
                                break
                            M = M_7 * (11**e)
                            if M > max_M:
                                break
                            candidates.append(M)
        
        if time.time() > deadline - 25:
            return
        
        # For each candidate M, find all k such that k*sigma(k) = M
        # k must be a divisor of M (since k*sigma(k)=M implies k divides M)
        # Actually, k doesn't have to divide M, but k*sigma(k)=M means k|M is necessary
        # because sigma(k) >= k+1 > k for k>1, so k < M, and k*sigma(k)=M
        # k divides M is NOT necessarily true. E.g., k=2, sigma(2)=3, M=6. 2|6 yes.
        # k=4, sigma(4)=7, M=28. 4|28 yes.
        # Actually k*sigma(k)=M, and sigma(k) is integer, so k|M.
        
        for M in candidates:
            if time.time() > deadline - 20:
                break
            
            # Find all divisors k of M such that k*sigma(k) = M
            # First find all divisors of M
            divisors = []
            temp = M
            i = 2
            factors = []
            while i * i <= temp:
                if temp % i == 0:
                    count = 0
                    while temp % i == 0:
                        temp //= i
                        count += 1
                    factors.append((i, count))
                i += 1
            if temp > 1:
                factors.append((temp, 1))
            
            # Generate all divisors
            divisors = [1]
            for p, exp in factors:
                new_divs = []
                for d in divisors:
                    power = 1
                    for _ in range(exp + 1):
                        new_divs.append(d * power)
                        power *= p
                divisors = new_divs
            
            # Filter: k <= isqrt(M) and k*sigma(k) == M
            valid_ks = []
            sqrt_M = math.isqrt(M)
            for k in divisors:
                if k > sqrt_M:
                    continue
                if k <= 0:
                    continue
                # Check k*sigma(k) == M
                sig = sigma_func(k)
                if k * sig == M:
                    valid_ks.append(k)
            
            if len(valid_ks) > len(best_ks):
                update_best(M, valid_ks)
        
        return

    def search_with_numpy(K):
        local_start = time.time()
        
        try:
            # 1. Sieve for sigma
            sigma = np.zeros(K + 1, dtype=np.int64)
            
            # Check time periodically to avoid exceeding deadline
            check_interval = max(1000, K // 500)  # Check at most ~500 times
            
            for i in range(1, K + 1):
                if i % check_interval == 0 and time.time() > deadline - 15:
                    return 1  # Time out
                
                sigma[i::i] += i
                
            if time.time() > deadline - 15:
                return 1
                
            # 2. Compute v(k) = k * sigma(k)
            ks = np.arange(1, K + 1, dtype=np.int64)
            vals = ks * sigma[1:K+1]
            
            # 3. Filter valid M
            max_M = 10**18
            valid_mask = vals <= max_M
            if not np.any(valid_mask):
                return 1
                
            vals = vals[valid_mask]
            ks = ks[valid_mask]
            
            if len(vals) < 2:
                return 1
                
            # 4. Sort and group
            order = np.argsort(vals)
            sorted_vals = vals[order]
            sorted_ks = ks[order]
            
            # Find consecutive equal values
            diffs = np.diff(sorted_vals)
            change_points = np.where(diffs > 0)[0] + 1
            
            start_points = np.concatenate(([0], change_points))
            end_points = np.concatenate((change_points, [len(sorted_vals)]))
            
            counts = end_points - start_points
            
            if len(counts) == 0:
                return 1
                
            max_idx = np.argmax(counts)
            max_count = counts[max_idx]
            
            if max_count > len(best_ks):
                s = start_points[max_idx]
                e = end_points[max_idx]
                M = int(sorted_vals[s])
                ks_group = sorted_ks[s:e]
                update_best(M, ks_group)
                
            return 0  # Success
            
        except MemoryError:
            return 2  # Memory error
        except Exception as e:
            print(f"Error in search: {e}")
            return 3  # Other error
    
    # First, try targeted M search (fast, no large memory)
    if time.time() < deadline - 40:
        try:
            search_targeted_M()
        except Exception as e:
            print(f"Targeted M search error: {e}")
    
    # Then try sieve-based search with remaining time
    # Try K values from largest to smallest
    target_Ks = [5000000, 2000000, 1000000, 500000]
    
    for K in target_Ks:
        if time.time() > deadline - 25:
            break
            
        status = search_with_numpy(K)
        elapsed = time.time() - start_time
        print(f"Search K={K} finished with status {status}. Total elapsed: {elapsed:.2f}s")
        
        if status == 0:
            # Success, we can try smaller K to see if we find better M
            # But smaller K means fewer k values, so unlikely to find more solutions
            # Actually, we want to maximize the count, so larger K is better
            # If we succeeded, that's our best shot from sieve
            break
        elif status == 2:
            # Memory error, try smaller K
            continue
        else:
            # Other error or timeout, try smaller K
            continue
    
    return {'M': best_M, 'k': best_ks}
