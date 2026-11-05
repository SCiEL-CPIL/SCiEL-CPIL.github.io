import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    N = 1000000
    # P[i] = powerful part of i
    P = np.ones(N + 1, dtype=np.int64)
    
    # Sieve for primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # Only primes p where p^2 <= N can contribute to powerful parts
    # So p <= sqrt(N) = 1000
    primes = primes[primes <= int(math.isqrt(N))]
    
    # For each prime, multiply p^k into P[m] for appropriate multiples
    for p in primes:
        if time.time() > deadline:
            break
        p = int(p)
        power = p * p  # p^2
        while power <= N:
            next_power = power * p  # p^(k+1)
            # For each multiple m of power, if p^(k+1) does not divide m, multiply P[m] by power
            # We need: m % power == 0 and m % next_power != 0
            # This means m is divisible by p^k but not p^(k+1)
            # i.e., the exact power of p dividing m is exactly k
            # So we want indices m where m = power * t and t % p != 0
            # m ranges over power, 2*power, 3*power, ... up to N
            # Filter: m % next_power != 0
            # Equivalent: (m // power) % p != 0
            
            # Create array of multiples
            multiples = np.arange(power, N + 1, power, dtype=np.int64)
            # Filter those where (m // power) % p != 0
            t_vals = multiples // power
            mask = (t_vals % p) != 0
            valid_multiples = multiples[mask]
            P[valid_multiples] *= power
            power = next_power
            if power > N:
                break
    
    # Find n in [10000, 1000000] that maximizes P[n] * P[n+1]
    # P is 1-indexed, so P[n] is the powerful part of n
    # Use vectorized numpy for speed
    products = P[10000:N] * P[10001:N + 1]
    best_idx = int(np.argmax(products))
    best_n = 10000 + best_idx
    best_val = products[best_idx]
    
    # CRT-based constructive search: generate high-density candidates
    # and verify they are consistent with the brute-force result
    if time.time() < deadline - 5:
        crt_candidates = set()
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        max_exp = 6
        for p in small_primes:
            for e in range(2, max_exp + 1):
                pe = p ** e
                if pe > N:
                    break
                for q in small_primes:
                    if q == p:
                        continue
                    for f in range(2, max_exp + 1):
                        qf = q ** f
                        if qf > N:
                            break
                        # Solve n ≡ 0 mod pe, n ≡ -1 mod qf
                        # n = pe * k, so pe*k ≡ -1 mod qf
                        # k ≡ -pe^(-1) mod qf
                        # Need gcd(pe, qf) = 1, which is true since p≠q
                        # Find inverse of pe mod qf
                        try:
                            inv = pow(pe, -1, qf)
                        except ValueError:
                            continue
                        k0 = (-inv) % qf
                        # n = pe * (k0 + t*qf) = pe*k0 + t*pe*qf
                        # n in [10000, 999999]
                        lcm_val = pe * qf
                        n0 = pe * k0
                        # Find smallest t such that n0 + t*lcm_val >= 10000
                        if n0 >= 10000:
                            t_min = 0
                        else:
                            t_min = (10000 - n0 + lcm_val - 1) // lcm_val
                        t_max = (999999 - n0) // lcm_val
                        if t_max >= t_min:
                            for t in range(t_min, t_max + 1):
                                n_cand = n0 + t * lcm_val
                                if 10000 <= n_cand <= 999999:
                                    crt_candidates.add(n_cand)
        
        # Evaluate CRT candidates
        for n_c in crt_candidates:
            if P[n_c] * P[n_c + 1] > best_val:
                best_val = P[n_c] * P[n_c + 1]
                best_n = n_c
    
    return int(best_n)
