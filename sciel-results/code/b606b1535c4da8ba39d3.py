import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    def elapsed():
        return time.time() - start_time

    # Phase 1: Sieve to find initial candidates M
    # We sieve up to N1 to find M values.
    N1 = 2000000
    
    # Sieve for smallest prime factor
    spf = list(range(N1 + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(N1)) + 1):
        if spf[i] == i:
            for j in range(i * i, N1 + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Compute sigma(k) and f(k) = k * sigma(k)
    sigma = [0] * (N1 + 1)
    sigma[1] = 1
    f_vals = [0] * (N1 + 1)
    f_vals[1] = 1
    
    for k in range(2, N1 + 1):
        n = k
        result = 1
        while n > 1:
            p = spf[n]
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            result *= power_sum
        sigma[k] = result
        f_vals[k] = k * result

    # Bucket M = k * sigma(k)
    M_to_ks = defaultdict(list)
    for k in range(1, N1 + 1):
        M = f_vals[k]
        if M <= 10**18:
            M_to_ks[M].append(k)

    # Find candidates with count >= 2
    candidates = [(M, ks) for M, ks in M_to_ks.items() if len(ks) >= 2]
    candidates.sort(key=lambda x: len(x[1]), reverse=True)

    best_M = None
    best_ks = []

    if not candidates:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Phase 2: Expand top candidates by checking all divisors of M
    # We check the top 20 candidates to ensure we don't time out.
    top_candidates = candidates[:20]
    
    for M, initial_ks in top_candidates:
        if elapsed() > deadline - 10:
            break
            
        # Factorize M
        temp_M = M
        factors = {}
        d = 2
        while d * d <= temp_M:
            while temp_M % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp_M //= d
            d += 1 if d == 2 else 2
        if temp_M > 1:
            factors[temp_M] = factors.get(temp_M, 0) + 1
        
        # Generate all divisors of M
        divisors = [1]
        for p, exp in factors.items():
            new_divs = []
            p_powers = [1]
            for _ in range(exp):
                p_powers.append(p_powers[-1] * p)
            for d in divisors:
                for pp in p_powers:
                    new_divs.append(d * pp)
            divisors = new_divs
        
        # Filter and verify
        max_k = math.isqrt(M)
        verified_ks = set()
        
        for k in divisors:
            if k > max_k:
                continue
            # Compute sigma(k)
            # k divides M, so prime factors of k are in factors
            # We need to determine the exponent of each prime in k
            # Then compute sigma(k) = prod (p^(a+1)-1)/(p-1)
            
            # Efficiently compute sigma(k)
            n = k
            sig_k = 1
            # Since k <= 10^9 usually, and factors are known, we can just factor k using the known primes
            # Or simpler: just use the standard factorization method if k is small enough, 
            # but we already have the primes of M.
            
            # Let's factor k using the primes of M
            temp_k = k
            for p in factors:
                if temp_k % p == 0:
                    power_sum = 1
                    power = 1
                    while temp_k % p == 0:
                        temp_k //= p
                        power *= p
                        power_sum += power
                    sig_k *= power_sum
            if temp_k > 1:
                # This shouldn't happen if factors is complete for M
                # But just in case, handle large prime factor
                sig_k *= (1 + temp_k)
            
            if sig_k == M // k:
                verified_ks.add(k)
                
        if len(verified_ks) > len(best_ks):
            best_ks = list(verified_ks)
            best_M = M

    if best_M is None:
        best_M = candidates[0][0]
        best_ks = candidates[0][1]

    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}

if __name__ == '__main__':
    res = construct()
    print(res)
