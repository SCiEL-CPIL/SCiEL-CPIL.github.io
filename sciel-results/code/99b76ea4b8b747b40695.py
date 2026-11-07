import time
import math
from collections import defaultdict, Counter

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    max_m = 10**18

    # Determine a reasonable limit for k to search within time budget
    # Sieve of sigma(k) is O(N log N). N=5e6 is fast (<1s), N=2e7 is moderate (~2-3s).
    # We can afford a larger N. Let's aim for N=10^7.
    # However, M = k * sigma(k). For k=10^7, sigma(k) is roughly k log log k ~ 10^7 * 3 = 3*10^7.
    # M ~ 3*10^14, well within 10^18.
    # We can even go higher. Let's set N based on time.
    
    N = 10_000_000  # 10 million
    
    # Check if we have enough time for a larger N
    # Estimate: N=10^6 takes ~0.1s. N=10^7 takes ~1-2s. N=5*10^7 takes ~10s.
    # Let's stick to 10^7 to be safe, or dynamically adjust.
    # For now, fixed 10^7 is a good balance.
    
    # Sieve for sigma(k)
    # sigma[1] = 1
    # For prime p, sigma(p^e) = 1 + p + ... + p^e
    # We can use a linear sieve or a standard sieve.
    # Standard sieve:
    # Initialize sigma = [1] * (N+1)
    # For i from 2 to N:
    #   If sigma[i] == 1, i is prime.
    #   For multiple j of i:
    #      This is tricky for multiplicative functions.
    
    # Better approach: 
    # 1. Sieve smallest prime factor (SPF) or just primes.
    # 2. Use the property: sigma(n) = sigma(p^a * m) = sigma(p^a) * sigma(m) if gcd(p,m)=1.
    # Or simply:
    # sigma = [1] * (N+1)
    # For i in range(2, N+1):
    #    if is_prime[i]:
    #       for j in range(i, N+1, i):
    #          # j is divisible by i
    #          # We need to add i * (1 + i + ... + i^k) to sigma[j]? No.
    #          # Actually, a simpler sieve for sigma:
    #          # sigma(n) = sum_{d|n} d
    #          # We can do: for i in 1..N: for j in i, 2i, 3i...: sigma[j] += i
    #          # This is O(N log N). For N=10^7, sum(1/i) ~ log N ~ 16.
    #          # 10^7 * 16 = 1.6 * 10^8 operations. In Python, this might be too slow (~5-10s).
    #          # Let's optimize.
    
    # Optimized Sieve for Sigma:
    # We can compute sigma using the factorization.
    # 1. Find smallest prime factor (SPF) for all numbers up to N.
    # 2. Compute sigma[n] using SPF.
    
    # Step 1: SPF Sieve
    spf = list(range(N + 1))  # smallest prime factor
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Compute sigma[n]
    # sigma[n] = sigma[p^a * m] = sigma[p^a] * sigma[m]
    # We can compute this iteratively.
    sigma = [1] * (N + 1)
    # We need to track the power of the smallest prime factor.
    # Let n = p^a * m, where p = spf[n].
    # sigma[n] = sigma[m] * (1 + p + ... + p^a)
    # We can store p^a and the sum.
    
    # Arrays to store p^a and sum(1..p^a) for the smallest prime part
    # Or simpler:
    # sigma[1] = 1
    # For n from 2 to N:
    #   p = spf[n]
    #   m = n // p
    #   if m % p == 0:
    #       # n has p^a where a > 1
    #       # Let m = p^(a-1) * k. Then n = p^a * k.
    #       # sigma[n] = sigma[k] * (1 + p + ... + p^a)
    #       # sigma[m] = sigma[k] * (1 + p + ... + p^(a-1))
    #       # This requires knowing sigma[k].
    
    # Alternative simple method for sigma:
    # sigma = [0] * (N+1)
    # for i in range(1, N+1):
    #     for j in range(i, N+1, i):
    #         sigma[j] += i
    # This is O(N log N). 
    # In Python, nested loops are slow. 
    # Let's try to vectorize or use a faster logic.
    
    # Actually, let's stick to the recursive generator but optimize it?
    # No, the sieve is conceptually better if we can make it fast.
    
    # Let's use the divisor sum sieve but with a limit on time.
    # If N=10^6, it's fast. N=10^7 might be borderline.
    # Let's try N=2_000_000 first to ensure speed, then expand if time permits.
    
    # Revised Plan:
    # 1. Run sieve for N=2_000_000.
    # 2. Check time. If < 20s, run again for N=10_000_000? No, just pick a good N.
    # 3. N=5_000_000 is a good target.
    
    N = 5_000_000
    
    # Sieve for sigma
    sigma = [0] * (N + 1)
    # This loop is O(N log N)
    # For i in 1..N:
    #    for j in i, 2i...:
    #        sigma[j] += i
    
    # Optimization: Use NumPy if available? Yes.
    try:
        import numpy as np
        sigma = np.zeros(N + 1, dtype=np.int64)
        # Loop over i
        # for i in range(1, N+1):
        #     sigma[i::i] += i
        # This is still a Python loop over N.
        # But the inner slice operation is C-speed.
        # N=5e6, log N ~ 15. Total slices ~ 5e6 * 15 = 75e6 slice ops? No.
        # The loop runs N times. Each slice op is O(N/i).
        # Total work is sum(N/i) = N log N.
        # But the overhead of 5e6 Python loop iterations is significant.
        
        # Let's try a pure Python loop first, it might be faster due to less overhead if N is small enough?
        # No, NumPy slice addition is much faster per element.
        
        # Let's use the NumPy approach.
        for i in range(1, N + 1):
            if time.time() > deadline - 30:
                break
            sigma[i::i] += i
    except ImportError:
        # Fallback to pure Python if NumPy fails (unlikely)
        sigma = [0] * (N + 1)
        for i in range(1, N + 1):
            if time.time() > deadline - 30:
                break
            for j in range(i, N + 1, i):
                sigma[j] += i

    # Now we have sigma[k] for all k <= N.
    # Generate M = k * sigma[k]
    m_counts = defaultdict(int)
    m_examples = defaultdict(list)
    
    # We want to find M with many solutions.
    # Iterate k from 1 to N
    # M = k * sigma[k]
    # If M <= 10^18, count it.
    
    # To speed up, we can just loop.
    # N=5e6 loop is fast.
    
    for k in range(1, N + 1):
        if time.time() > deadline - 20:
            break
        s = sigma[k]
        m_val = k * s
        if m_val <= max_m:
            m_counts[m_val] += 1
            if len(m_examples[m_val]) < 10:
                m_examples[m_val].append(k)
        else:
            # Since k is increasing and sigma(k) is generally increasing,
            # M = k*sigma(k) is not strictly increasing, but tends to increase.
            # We can't break easily because sigma(k) can drop (e.g., prime vs composite).
            # So we just continue.
            pass

    if not m_counts:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Sort by count
    sorted_ms = sorted(m_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Factorization utilities
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            x = 2
            y = 2
            d = 1
            c = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            c += 1

    def factorize(n):
        factors = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
        else:
            stack = [n]
            while stack:
                x = stack.pop()
                if x == 1:
                    continue
                if is_prime(x):
                    factors.append(x)
                else:
                    d = pollard_rho(x)
                    stack.append(d)
                    stack.append(x // d)
        return sorted(factors)

    def generate_divisors_from_factors(factors):
        c = Counter(factors)
        primes_list = sorted(c.keys())
        exponents_list = [c[p] for p in primes_list]

        divs = [1]
        for i in range(len(primes_list)):
            p = primes_list[i]
            e = exponents_list[i]
            current_divs = []
            power = 1
            for k in range(e + 1):
                for d in divs:
                    current_divs.append(d * power)
                power *= p
            divs = current_divs
        return divs

    def find_all_k_for_M(M):
        if M < 2:
            return []

        factors = factorize(M)
        divs = generate_divisors_from_factors(factors)

        m_primes = sorted(set(factors))
        valid_ks = []
        sqrt_m = math.isqrt(M)

        for d in divs:
            if d > sqrt_m:
                continue
            if d < 1:
                continue

            temp_d = d
            sigma_d = 1
            for p in m_primes:
                if temp_d % p == 0:
                    e = 0
                    while temp_d % p == 0:
                        temp_d //= p
                        e += 1
                    sigma_d *= (p ** (e + 1) - 1) // (p - 1)
            if temp_d > 1:
                temp_factors = factorize(temp_d)
                temp_c = Counter(temp_factors)
                for p, e in temp_c.items():
                    sigma_d *= (p ** (e + 1) - 1) // (p - 1)

            if d * sigma_d == M:
                valid_ks.append(d)

        return valid_ks

    # Try top candidates
    best_m = 0
    best_k_list = []
    
    num_candidates = min(50, len(sorted_ms))
    
    for i in range(num_candidates):
        if time.time() > deadline - 5:
            break
        
        M, count = sorted_ms[i]
        
        # Optimization: If the current best count is already >= this candidate's count, 
        # we can skip if we are sure we won't find more? 
        # No, we need to find ALL k for this M. 
        # But if count < len(best_k_list), we can't beat it.
        if count < len(best_k_list):
            continue

        try:
            valid_ks = find_all_k_for_M(M)
            if len(valid_ks) > len(best_k_list):
                best_k_list = valid_ks
                best_m = M
        except Exception:
            continue

    if not best_k_list:
        best_m, count = sorted_ms[0]
        best_k_list = m_examples[best_m]

    # Final validation
    distinct_k = list(set(best_k_list))
    sqrt_m = math.isqrt(best_m)
    valid_k = [k for k in distinct_k if 1 <= k <= sqrt_m]
    valid_k.sort()

    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]

    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
