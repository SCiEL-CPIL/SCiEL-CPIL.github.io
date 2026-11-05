"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = start_time + 78  # safety margin before 90s kill
    N = 1000000
    MIN_N = 10000

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Multiplicative number-theoretic sieve to compute powerful part P(n) for all n in [1, N]
    #    P(n) = product of p^e for each prime p where e = v_p(n) >= 2
    P = [1] * (N + 1)
    
    # Find all primes up to N
    primes = []
    for i in range(2, N + 1):
        if spf[i] == i:
            primes.append(i)
    
    for p in primes:
        pk = p * p
        while pk <= N:
            p_next = pk * p
            # For each multiple m of pk, if m is not a multiple of p_next,
            # then v_p(m) = k, so we multiply p^k = pk into P[m].
            for m in range(pk, N + 1, pk):
                if m % p_next != 0:
                    P[m] *= pk
            pk = p_next
    
    # 3. Find n in [MIN_N, N-1] that maximizes P[n] * P[n+1]
    #    Note: n=N is not evaluated because P[N+1] is out of bounds, 
    #    but P(N)*P(N+1) is small (P(N+1)=1 for N=10^6) and thus not the maximum.
    
    best_n = MIN_N
    best_val = 0
    
    for n in range(MIN_N, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    # 4. CRT-based constructive search to verify and potentially find better candidates
    #    Generate n such that n ≡ 0 (mod p^e) and n+1 ≡ 0 (mod q^f) for distinct primes
    #    This ensures P[n] >= p^e and P[n+1] >= q^f
    
    if time.time() < DEADLINE - 10:
        # Generate all prime powers p^e <= N with e >= 2
        prime_powers = []  # list of (p, e, p^e)
        for p in primes:
            pe = p * p
            e = 2
            while pe <= N:
                prime_powers.append((p, e, pe))
                pe *= p
                e += 1
        
        # For pairs of prime powers from disjoint prime sets, solve CRT
        # n ≡ 0 (mod a), n ≡ -1 (mod b) where gcd(a,b)=1
        # n = a * k, a*k ≡ -1 (mod b) => k ≡ -a^{-1} (mod b)
        # n = a * k0 + a*b * t
        
        # Collect candidate n values from CRT
        crt_candidates = set()
        
        # Sort prime powers by value descending to prioritize large ones
        prime_powers.sort(key=lambda x: -x[2])
        
        # Limit to top prime powers for efficiency
        max_pp = min(len(prime_powers), 200)
        pp_list = prime_powers[:max_pp]
        
        for i in range(len(pp_list)):
            if time.time() > DEADLINE - 5:
                break
            p1, e1, a = pp_list[i]
            for j in range(len(pp_list)):
                if i == j:
                    continue
                p2, e2, b = pp_list[j]
                if p1 == p2:
                    continue  # need distinct primes
                mod = a * b
                if mod > N + 1:
                    continue
                # Solve n ≡ 0 (mod a), n ≡ -1 (mod b)
                # Find inverse of a mod b
                try:
                    inv = pow(a, -1, b)
                except ValueError:
                    continue
                k0 = (-inv) % b
                n0 = a * k0
                # n = n0 + mod * t for t >= 0
                # Find t such that MIN_N <= n <= N-1
                if n0 >= MIN_N and n0 <= N - 1:
                    crt_candidates.add(n0)
                # Next solution
                if n0 < MIN_N:
                    t_start = (MIN_N - n0 + mod - 1) // mod
                else:
                    t_start = 0
                t = t_start
                n = n0 + mod * t
                while n <= N - 1:
                    crt_candidates.add(n)
                    t += 1
                    n = n0 + mod * t
                    if n > N - 1:
                        break
                # Also check if we can go backwards
                if n0 > N - 1:
                    t_end = (N - 1 - n0) // mod
                    if t_end < 0:
                        t_end = 0
                    for tt in range(t_end + 1):
                        nn = n0 + mod * tt
                        if MIN_N <= nn <= N - 1:
                            crt_candidates.add(nn)
        
        # Evaluate CRT candidates
        for n in crt_candidates:
            if MIN_N <= n <= N - 1:
                val = P[n] * P[n + 1]
                if val > best_val:
                    best_val = val
                    best_n = n
        
        # 5. Additional: check neighbors of top CRT candidates for nearby high values
        if time.time() < DEADLINE - 2:
            # Get top 50 CRT candidates by P[n]*P[n+1]
            crt_vals = []
            for n in crt_candidates:
                if MIN_N <= n <= N - 1:
                    crt_vals.append((P[n] * P[n + 1], n))
            crt_vals.sort(reverse=True)
            for _, n in crt_vals[:50]:
                for dn in range(-20, 21):
                    nn = n + dn
                    if MIN_N <= nn <= N - 1:
                        val = P[nn] * P[nn + 1]
                        if val > best_val:
                            best_val = val
                            best_n = nn
    
    return best_n

# EVOLVE-BLOCK-END
