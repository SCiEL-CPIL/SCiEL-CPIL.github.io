import time
import math

def construct():
    deadline = time.time() + 55.0
    answers = [None] * 254
    
    # Precompute smallest prime factor (SPF) for fast factorization
    # B = n * x. n <= 256. If x up to 5000, B <= 1,280,000
    # Let's set LIMIT to 2,000,000 to be safe
    LIMIT = 2_000_000
    spf = list(range(LIMIT + 1))
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Factorize num using precomputed SPF. Returns list of (prime, exponent)."""
        if num <= 1:
            return []
        factors = []
        while num > 1:
            p = spf[num]
            cnt = 0
            while num % p == 0:
                num //= p
                cnt += 1
            factors.append((p, cnt))
        return factors
    
    def generate_divisors_of_square(factors):
        """
        Given factors of B, generate all divisors of B^2.
        If B = p1^a1 * p2^a2, then B^2 = p1^(2a1) * p2^(2a2).
        Divisors are p1^k1 * p2^k2 where 0 <= k1 <= 2a1, etc.
        """
        divs = [1]
        for p, exp in factors:
            max_exp = 2 * exp
            new_divs = []
            for d in divs:
                pk = 1
                for k in range(max_exp + 1):
                    new_divs.append(d * pk)
                    pk *= p
            divs = new_divs
        return divs

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            for j in range(idx, 254):
                answers[j] = None
            break
            
        found = None
        x_min = n // 4 + 1
        
        # Phase 1: Try small x range (up to 50 attempts)
        max_x_attempts_phase1 = 50
        for i in range(max_x_attempts_phase1):
            x = x_min + i
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            if B > LIMIT:
                continue
                
            # We need d such that:
            # 1. d | B^2
            # 2. (d + B) % A == 0
            # 3. (B^2 // d + B) % A == 0
            # 4. y = (d+B)//A > x
            # 5. z = (B^2//d + B)//A > y
            
            # Condition 4: d + B > x*A => d > x*A - B = x(4x-n) - nx = 4x^2 - 2nx
            d_lower_bound = 4 * x * x - 2 * n * x
            
            # Condition 5: z > y => B^2/d > d => d^2 < B^2 => d < B
            if d_lower_bound >= B:
                continue
                
            factors = factorize(B)
            divs = generate_divisors_of_square(factors)
            
            # Sort and filter divisors in (d_lower_bound, B)
            divs.sort()
            
            for d in divs:
                if d <= d_lower_bound:
                    continue
                if d >= B:
                    break
                    
                num_y = d + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                if y <= x:
                    continue
                    
                num_z = (B * B) // d + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                if z <= y:
                    continue
                
                if z <= 1000000000000:
                    found = [x, y, z]
                    break
                    
            if found:
                break
                
        # Phase 2: If not found, try larger x with fewer attempts but wider range
        if not found:
            max_x_attempts_phase2 = 200
            for i in range(max_x_attempts_phase1, max_x_attempts_phase1 + max_x_attempts_phase2):
                x = x_min + i
                if time.time() > deadline:
                    break
                    
                A = 4 * x - n
                if A <= 0:
                    continue
                B = n * x
                
                if B > LIMIT:
                    continue
                    
                d_lower_bound = 4 * x * x - 2 * n * x
                if d_lower_bound >= B:
                    continue
                    
                factors = factorize(B)
                divs = generate_divisors_of_square(factors)
                divs.sort()
                
                for d in divs:
                    if d <= d_lower_bound:
                        continue
                    if d >= B:
                        break
                        
                    num_y = d + B
                    if num_y % A != 0:
                        continue
                    y = num_y // A
                    
                    if y <= x:
                        continue
                        
                    num_z = (B * B) // d + B
                    if num_z % A != 0:
                        continue
                    z = num_z // A
                    
                    if z <= y:
                        continue
                    
                    if z <= 1000000000000:
                        found = [x, y, z]
                        break
                        
                if found:
                    break
                    
        answers[idx] = found
        
    return answers
