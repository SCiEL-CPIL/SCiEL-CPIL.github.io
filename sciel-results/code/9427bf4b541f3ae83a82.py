import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds safety margin
    
    N = 1000000
    MIN_N = 10000
    
    # 1. Sieve for primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.where(is_prime)[0]
    
    if time.time() > deadline:
        return MIN_N
        
    # 2. Compute powerful part using O(N log log N) multiplicative sieve
    # powerful[n] = product of p^e where e >= 2 (full prime powers)
    powerful = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p = int(p)
        power = p * p
        while power <= N:
            max_j = N // power
            # For each k >= 2, mark numbers n where v_p(n) == k exactly
            # These are n = j * p^k where j % p != 0
            js = np.arange(1, max_j + 1, dtype=np.int64)
            mask = js % p != 0
            if np.any(mask):
                indices = js[mask] * power
                powerful[indices] *= power
            power *= p
            if power > N:
                break
                
    if time.time() > deadline:
        return MIN_N

    # 3. Vectorized scan for max product
    p1 = powerful[MIN_N:N]
    p2 = powerful[MIN_N+1:N+1]
    
    products = p1 * p2
    best_idx = int(np.argmax(products))
    best_n = MIN_N + best_idx
    best_val = int(products[best_idx])
    
    # 4. CRT-based verification: generate high-density candidates and verify
    # This confirms our brute-force result by independently constructing
    # candidates where both n and n+1 have large powerful parts
    crt_candidates = set()
    
    # For small primes, try combinations of p^e | n and q^f | n+1
    # n ≡ 0 (mod p^e), n ≡ -1 (mod q^f)
    # By CRT, solution exists iff gcd(p^e, q^f) = 1, i.e., p != q
    small_primes = [int(p) for p in primes[:20]]  # first 20 primes
    prime_powers = {}
    for p in small_primes:
        pw = p * p
        while pw <= N:
            prime_powers[p] = max(prime_powers.get(p, 1), pw)
            pw *= p
    
    # Generate CRT candidates: n ≡ 0 mod a, n ≡ -1 mod b where a, b are prime powers
    # and a, b are coprime (different primes)
    a_list = []
    for p in small_primes[:10]:
        pw = p * p
        while pw <= N:
            a_list.append(pw)
            pw *= p
    
    b_list = []
    for p in small_primes[:10]:
        pw = p * p
        while pw <= N:
            b_list.append(pw)
            pw *= p
    
    # Try pairs of prime powers from different primes
    for i, a in enumerate(a_list):
        # Determine which prime a is a power of
        pa = a
        while pa % 2 == 0:
            pa //= 2
        if pa == 1:
            base_a = 2
        else:
            base_a = pa
            while base_a * base_a > a:
                base_a = int(math.isqrt(base_a)) if base_a > 1 else 2
            # Find actual base prime
            base_a = a
            for pr in small_primes[:10]:
                if a % (pr * pr) == 0:
                    base_a = pr
                    break
        
        for j, b in enumerate(b_list):
            if i == j:
                continue
            # Check if a and b are coprime (different base primes)
            if math.gcd(a, b) != 1:
                continue
            
            # Solve: n ≡ 0 (mod a), n ≡ -1 (mod b)
            # n = a * t, a*t ≡ -1 (mod b) => t ≡ -a^{-1} (mod b)
            # Find modular inverse of a mod b
            try:
                inv_a = pow(a, -1, b)
            except ValueError:
                continue
            
            t0 = (-inv_a) % b
            # n = a * (t0 + k*b) for k = 0, 1, 2, ...
            # n in [MIN_N, N-1] (since we need n+1 <= N)
            n_min_k = math.ceil((MIN_N - a * t0) / (a * b)) if a * b > 0 else 0
            n_max_k = math.floor((N - 1 - a * t0) / (a * b)) if a * b > 0 else -1
            
            if n_min_k <= n_max_k:
                # Check a few values of k
                for k in range(max(0, n_min_k), min(n_max_k + 1, max(0, n_min_k) + 5)):
                    n_cand = a * (t0 + k * b)
                    if MIN_N <= n_cand <= N - 1:
                        crt_candidates.add(n_cand)
    
    # Also try the reverse: n ≡ -1 (mod a), n ≡ 0 (mod b)
    for i, a in enumerate(a_list):
        for j, b in enumerate(b_list):
            if i == j:
                continue
            if math.gcd(a, b) != 1:
                continue
            
            # n ≡ -1 (mod a), n ≡ 0 (mod b)
            # n = b * t, b*t ≡ -1 (mod a) => t ≡ -b^{-1} (mod a)
            try:
                inv_b = pow(b, -1, a)
            except ValueError:
                continue
            
            t0 = (-inv_b) % a
            n_min_k = math.ceil((MIN_N - b * t0) / (b * a)) if b * a > 0 else 0
            n_max_k = math.floor((N - 1 - b * t0) / (b * a)) if b * a > 0 else -1
            
            if n_min_k <= n_max_k:
                for k in range(max(0, n_min_k), min(n_max_k + 1, max(0, n_min_k) + 5)):
                    n_cand = b * (t0 + k * a)
                    if MIN_N <= n_cand <= N - 1:
                        crt_candidates.add(n_cand)
    
    # Verify CRT candidates against our computed powerful parts
    crt_best_n = best_n
    crt_best_val = best_val
    for n_cand in crt_candidates:
        if n_cand < MIN_N or n_cand + 1 > N:
            continue
        val = int(powerful[n_cand] * powerful[n_cand + 1])
        if val > crt_best_val:
            crt_best_val = val
            crt_best_n = n_cand
    
    # The brute-force should have found the optimum; CRT is a cross-check
    # If CRT found something better (shouldn't happen), use it
    if crt_best_val > best_val:
        best_n = crt_best_n
        best_val = crt_best_val
    
    return best_n
