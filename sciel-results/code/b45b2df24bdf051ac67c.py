import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    # Primes 1 mod 4
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 811, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 991, 997]
    
    # Precompute a, b for each prime p = a^2 + b^2
    prime_reps = {}
    for p in p_1mod4:
        found = False
        for a in range(1, int(math.isqrt(p)) + 1):
            b2 = p - a*a
            b = int(math.isqrt(b2))
            if b*b == b2:
                prime_reps[p] = (a, b)
                found = True
                break
        if not found:
            # Should not happen for 1 mod 4 primes
            prime_reps[p] = (1, 1) # Fallback

    best_N = 0
    best_pairs = []
    
    # Try Q=1 (k=0) primarily, as it allows largest M
    # Q = 1
    # M <= 10^18
    
    M_max = 10**18
    
    # Find M with max number of representations
    # Maximize prod(e_i + 1)
    
    best_M = 1
    best_score = 0
    best_exponents = {}
    
    # Use DFS with pruning
    # We will use the first 30 primes to have enough options
    search_primes = p_1mod4[:30]
    
    def dfs(idx, current_M, current_score, prev_exp, exponents):
        nonlocal best_M, best_score, best_exponents
        
        if time_left() < 5:
            return
            
        if current_score > best_score:
            best_score = current_score
            best_M = current_M
            best_exponents = exponents.copy()
            
        if idx == len(search_primes):
            return
            
        p = search_primes[idx]
        
        # Upper bound on score if we take all remaining primes with exp 1
        # This is a loose bound, but helps pruning
        # Actually, just checking time is enough if we are fast
        
        max_e = prev_exp
        val = 1
        e = 0
        while current_M * val <= M_max:
            if e > max_e:
                break
            next_M = current_M * val
            next_score = current_score * (e + 1)
            
            exponents[p] = e
            dfs(idx + 1, next_M, next_score, e, exponents)
            del exponents[p]
            
            val *= p
            e += 1
            if val > M_max:
                break

    dfs(0, 1, 1, 100, {})
    
    M = best_M
    N = M # Q=1
    
    # Generate pairs for M
    # Factorization of M is in best_exponents
    # reps = [(1, 0)]
    reps = [(1, 0)]
    
    for p, e in best_exponents.items():
        if e == 0:
            continue
        a, b = prime_reps[p]
        # Generate reps for p^e
        # z = a + bi
        # reps_p = [z^k * conj(z)^(e-k) for k in 0..e]
        # Note: z^k * conj(z)^(e-k) = z^k * (a-bi)^(e-k)
        # This is equivalent to z^k * (z*conj(z))^(e-k) / z^(e-k)? No.
        # conj(z) = a - bi.
        # Let's just compute z^k and conj(z)^(e-k) and multiply.
        
        new_reps = []
        # Precompute powers of z and conj(z)
        pow_z = [1 + 0j] * (e + 1)
        pow_cz = [1 + 0j] * (e + 1)
        pow_z[0] = 1
        pow_cz[0] = 1
        z = complex(a, b)
        cz = complex(a, -b)
        for i in range(1, e + 1):
            pow_z[i] = pow_z[i-1] * z
            pow_cz[i] = pow_cz[i-1] * cz
            
        for k in range(e + 1):
            # Term is z^k * conj(z)^(e-k)
            term = pow_z[k] * pow_cz[e-k]
            # term is a complex number x + iy
            # This gives one representation of p^e
            # But we need to combine with existing reps
            # The set of reps for p^e is { z^k * conj(z)^(e-k) | k=0..e }
            # Wait, is this the full set?
            # Yes, for prime powers.
            
            # Now combine with current reps
            # Current reps are sums of squares for previous primes
            # New reps = { r1 + i r2 * (x + iy) | r1+ir2 in reps, x+iy in reps_p }
            # This is convolution in Z[i]
            
            # We can do this by iterating
            # But we can also just collect all candidate (x,y) for p^e first
            pass
        
        # Let's refine the combination step
        # reps_p_list = [ (real(part of z^k * cz^(e-k)), imag(...)) for k in range(e+1) ]
        reps_p_list = []
        for k in range(e + 1):
            term = pow_z[k] * pow_cz[e-k]
            reps_p_list.append((term.real, term.imag))
            
        # Combine reps and reps_p_list
        new_reps_combined = []
        for x1, y1 in reps:
            for x2, y2 in reps_p_list:
                # (x1 + i y1) * (x2 + i y2) = (x1x2 - y1y2) + i (x1y2 + y1x2)
                nx = x1 * x2 - y1 * y2
                ny = x1 * y2 + y1 * x2
                new_reps_combined.append((nx, ny))
        
        reps = new_reps_combined
        
    # Filter and format pairs
    pairs = []
    seen = set()
    for x, y in reps:
        # We want x^2 + y^2 = M
        # x, y can be negative
        # We want a = x^2, b = y^2
        # a <= b
        # x^2, y^2 are non-negative
        a = int(x**2)
        b = int(y**2)
        if a + b == M:
            if a > b:
                a, b = b, a
            if a >= 1 and b >= 1: # 0 is not powerful
                if (a, b) not in seen:
                    seen.add((a, b))
                    pairs.append([a, b])
                    
    if len(pairs) > len(best_pairs):
        best_pairs = pairs
        best_N = N

    # Fallback
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
