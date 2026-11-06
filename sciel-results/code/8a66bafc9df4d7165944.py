import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
        
    def find_sum_sq_rep(p):
        limit = int(math.isqrt(p)) + 1
        for a in range(1, limit):
            b2 = p - a*a
            b = math.isqrt(b2)
            if b*b == b2:
                return (a, b)
        return (1, 1)
        
    # Primes 1 mod 4
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97]
    
    reps = {}
    for p in primes_1mod4:
        reps[p] = find_sum_sq_rep(p)
        
    # Define exponents
    # We want to maximize pairs.
    # N = 5^2 * 13 * 17 * ... * 97
    exponents = {5: 2, 13: 1, 17: 1, 29: 1, 37: 1, 41: 1, 53: 1, 61: 1, 73: 1, 89: 1, 97: 1}
    
    N = 1
    for p, e in exponents.items():
        N *= p ** e
        
    if N > 10**18:
        # Fallback to smaller N if calculation is wrong or overflow
        # But we estimated 5*10^17
        pass
        
    # Generate Gaussian factors for each prime power
    # For p^e, factors are prod_{j=0}^e (a+bi)^j (a-bi)^(e-j)
    # This is equivalent to summing j=0 to e of C(e,j) (a+bi)^j (a-bi)^(e-j) ? No.
    # We just need the specific Gaussian integers.
    # Since e is small (1 or 2), we can just compute them.
    
    prime_factors = []
    for p, e in exponents.items():
        a, b = reps[p]
        g_plus = (a, b)
        g_minus = (a, -b)
        
        factors_for_p = []
        if e == 1:
            factors_for_p = [g_plus, g_minus]
        elif e == 2:
            # (a+bi)^2, (a+bi)(a-bi), (a-bi)^2
            # (a+bi)^2 = (a^2-b^2) + 2ab i
            r1 = a*a - b*b
            i1 = 2*a*b
            # (a+bi)(a-bi) = a^2+b^2 = p
            r2 = p
            i2 = 0
            # (a-bi)^2 = (a^2-b^2) - 2ab i
            r3 = a*a - b*b
            i3 = -2*a*b
            factors_for_p = [(r1, i1), (r2, i2), (r3, i3)]
        else:
            # General case if needed, but we only use 1 and 2
            # Use recursive multiplication
            curr = [(1, 0)]
            for _ in range(e):
                new_curr = []
                for (r, i) in curr:
                    # mult by g_plus
                    r1 = r*a - i*b
                    i1 = r*b + i*a
                    new_curr.append((r1, i1))
                    # mult by g_minus
                    r2 = r*a + i*b
                    i2 = r*b - i*a
                    new_curr.append((r2, i2))
                curr = new_curr
            # Note: This generates 2^e factors, but some might be duplicates?
            # For e=2, it generates 4 factors: (3,4), (5,0), (5,0), (3,-4).
            # We should unique them.
            seen = set()
            unique_factors = []
            for f in curr:
                if f not in seen:
                    seen.add(f)
                    unique_factors.append(f)
            factors_for_p = unique_factors
            
        prime_factors.append(factors_for_p)
        
    # Combine all factors
    # Start with [(1,0)]
    states = [(1, 0)]
    
    for factors in prime_factors:
        new_states = []
        for (r, i) in states:
            for (fr, fi) in factors:
                # Multiply (r+ii) by (fr+fi i)
                nr = r*fr - i*fi
                ni = r*fi + i*fr
                new_states.append((nr, ni))
        states = new_states
        
        if time_left() < 5:
            break
            
    # Extract pairs
    seen_pairs = set()
    pairs = []
    
    for (r, i) in states:
        x = abs(r)
        y = abs(i)
        a = x*x
        b = y*y
        if a > b:
            a, b = b, a
        # Check if a+b == N
        if a + b != N:
            continue
        if (a, b) not in seen_pairs:
            seen_pairs.add((a, b))
            pairs.append([a, b])
            
    # Truncate to 1000
    if len(pairs) > 1000:
        pairs = pairs[:1000]
        
    if not pairs:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    return {'N': N, 'pairs': pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
