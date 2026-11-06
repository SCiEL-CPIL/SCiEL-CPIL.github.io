import math
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Primes p ≡ 1 mod 4
    # We only need small ones.
    # 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257]
    
    # Precompute representations a, b for a^2 + b^2 = p
    # For small primes, we can find them quickly.
    p_to_rep = {}
    for p in primes_1mod4:
        found = False
        for a in range(1, int(math.isqrt(p)) + 1):
            b2 = p - a*a
            b = int(math.isqrt(b2))
            if b*b == b2:
                p_to_rep[p] = (a, b)
                found = True
                break
        if not found:
            # Should not happen for p ≡ 1 mod 4
            p_to_rep[p] = (1, 0) # Fallback, shouldn't be used

    max_val = 10**18
    best_M = 1
    best_K = 1
    best_exponents = {}
    
    # DFS to find optimal exponents
    # We limit the depth to the first ~15 primes to keep it fast.
    # The product of the first 12 primes is already large.
    search_primes = primes_1mod4[:15]
    
    def dfs(idx, current_M, current_K, exponents):
        nonlocal best_M, best_K, best_exponents
        
        if time.time() > deadline - 20.0:
            return

        if current_K > best_K:
            best_K = current_K
            best_M = current_M
            best_exponents = dict(exponents)
            
        if idx >= len(search_primes):
            return
            
        p = search_primes[idx]
        
        # Try exponents 0, 1, ...
        val = current_M
        exp = 0
        while val <= max_val:
            exponents[p] = exp
            next_K = current_K * (exp + 1)
            
            # Pruning: if next_K * (2 ** (len(search_primes) - idx - 1)) <= best_K, 
            # we can't beat best_K even if all remaining primes have exponent 1 (factor 2 each).
            # This is a loose bound but helps.
            max_possible_K = next_K * (2 ** (len(search_primes) - idx - 1))
            if max_possible_K <= best_K:
                break
                
            dfs(idx + 1, val, next_K, exponents)
            
            exp += 1
            val = val * p
            if val > max_val:
                break
                
        if p in exponents and exponents[p] == 0:
            del exponents[p]
            
    dfs(0, 1, 1, {})
    
    # If DFS didn't find a good M (e.g. M=1), fallback
    if best_M == 1:
         # Try a known good number
         # N = 5^2 * 13^2 * ...
         # Let's just use the first few primes with exp 1
         M_candidate = 1
         exps_candidate = {}
         for p in primes_1mod4[:10]:
             if M_candidate * p <= 10**18:
                 M_candidate *= p
                 exps_candidate[p] = 1
             else:
                 break
         if M_candidate > 1:
             best_M = M_candidate
             best_exponents = exps_candidate
             best_K = 2 ** len(exps_candidate)

    # Generate representations
    used_primes = [p for p in search_primes if p in best_exponents and best_exponents[p] > 0]
    
    if not used_primes:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Map prime to (a, b)
    # We need to be consistent with the sign? No, we take abs at the end.
    # But we need the correct Gaussian primes.
    # For p = a^2 + b^2, the Gaussian primes are a+bi and a-bi.
    
    # Get exponents
    exps = [best_exponents[p] for p in used_primes]
    
    # Generate all combinations of k_i
    # Number of combinations is best_K
    pairs_set = set()
    
    # Ranges for k_i
    ranges = [range(e + 1) for e in exps]
    
    def complex_mul(c1, c2):
        re1, im1 = c1
        re2, im2 = c2
        return (re1*re2 - im1*im2, re1*im2 + im1*re2)
        
    def complex_pow(c, n):
        if n == 0:
            return (1, 0)
        res = (1, 0)
        base = c
        while n > 0:
            if n % 2 == 1:
                res = complex_mul(res, base)
            base = complex_mul(base, base)
            n //= 2
        return res

    # Precompute powers of pi and pi_bar for each prime to speed up loop
    # pi_i = (a_i, b_i), pi_bar_i = (a_i, -b_i)
    # We need pi_i^k for k=0..e_i and pi_bar_i^(e_i-k)
    
    prime_powers = []
    for i, p in enumerate(used_primes):
        a, b = p_to_rep[p]
        pi = (a, b)
        pi_bar = (a, -b)
        e = exps[i]
        
        powers_pi = [(1, 0)]
        for k in range(1, e+1):
            powers_pi.append(complex_mul(powers_pi[-1], pi))
            
        powers_pi_bar = [(1, 0)]
        for k in range(1, e+1):
            powers_pi_bar.append(complex_mul(powers_pi_bar[-1], pi_bar))
            
        prime_powers.append((powers_pi, powers_pi_bar))
        
    for ks in itertools.product(*ranges):
        # z = prod (a_i + i b_i)^{k_i} * (a_i - i b_i)^{e_i - k_i}
        z = (1, 0)
        for i, p in enumerate(used_primes):
            k = ks[i]
            e = exps[i]
            powers_pi, powers_pi_bar = prime_powers[i]
            
            part1 = powers_pi[k]
            part2 = powers_pi_bar[e - k]
            
            z = complex_mul(z, part1)
            z = complex_mul(z, part2)
            
        u, v = z
        u = abs(u)
        v = abs(v)
        
        if u == 0 or v == 0:
            continue
            
        a = u*u
        b = v*v
        
        if a > b:
            a, b = b, a
            
        pairs_set.add((a, b))
        
        if time.time() > deadline - 10.0:
            break

    if not pairs_set:
        return {'N': 2, 'pairs': [[1, 1]]}

    pairs = sorted(list(pairs_set))
    N = best_M
    
    # Final verification is skipped for speed, but we trust the construction.
    # However, we should ensure N <= 10^18 and N >= 2.
    if N < 2 or N > 10**18:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': N, 'pairs': [[a, b] for a, b in pairs]}
