import time

def construct():
    answers = []
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precompute smallest prime factor (SPF) up to max B
    max_B = 256 * 192 + 100
    spf = list(range(max_B + 1))
    for i in range(2, int(max_B**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_B + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Factorize using SPF table."""
        factors = {}
        while num > 1:
            p = spf[num]
            cnt = 0
            while num % p == 0:
                num //= p
                cnt += 1
            factors[p] = cnt
        return factors
    
    factor_cache = {}
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # Tighten search range: for distinct x<y<z, we need x < n/4 (since 1/x > 4/n/3 => x < 3n/4)
        # But also from 1/x + 1/y + 1/z = 4/n with x<y<z, we have 3/x > 4/n => x < 3n/4
        # And 1/x < 4/n => x > n/4
        # More tightly: since y > x, 1/y < 1/x, so 1/x + 1/y + 1/z < 3/x, giving x < 3n/4
        # Also 1/x + 1/y + 1/z > 1/x, giving x > n/4
        # For distinct positive integers, try x from floor(n/4)+1 to floor(3n/4)
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue
            
            B_sq = B * B
            neg_B_mod_A = (-B) % A
            
            if B in factor_cache:
                factors = factor_cache[B]
            else:
                factors = factorize(B)
                factor_cache[B] = factors
            
            # Generate divisors of B^2, but only up to B (since u and v=B^2/u, we can assume u<=B)
            divs = [1]
            for p, e in factors.items():
                new_divs = []
                pe = 2 * e
                pow_p = 1
                for i in range(pe + 1):
                    for d in divs:
                        val = d * pow_p
                        if val <= B:  # Only keep divisors <= B to reduce list size
                            new_divs.append(val)
                    pow_p *= p
                divs = new_divs
            
            found_sol = None
            for u in divs:
                if u % A != neg_B_mod_A:
                    continue
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                v = B_sq // u
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                if x == y or x == z or y == z:
                    continue
                
                if y > 10**12 or z > 10**12:
                    continue
                
                found_sol = [x, y, z]
                break
            
            if found_sol:
                found = found_sol
                break
        
        answers.append(found)
        
    return answers
