import time
import math
import bisect

def construct():
    deadline = time.time() + 55.0
    answers = [None] * 254
    MAX_BOUND = 10**12
    
    # Precompute Smallest Prime Factor (SPF)
    LIMIT = 1500000
    spf = list(range(LIMIT + 1))
    for i in range(2, int(LIMIT**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(n):
        factors = {}
        while n > 1:
            p = spf[n]
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            factors[p] = count
        return factors

    def get_sorted_divisors_sq(B):
        fac_B = factorize(B)
        divs = [1]
        for p, e in fac_B.items():
            exp = 2 * e
            current_divs = []
            for d in divs:
                val = d
                for _ in range(exp + 1):
                    current_divs.append(val)
                    val *= p
            divs = current_divs
        divs.sort()
        return divs

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        x_max = 20 * n + 100
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            B_sq = B * B
            
            # Bounds for u
            # y = (u+B)/A <= MAX_BOUND => u <= A*MAX_BOUND - B
            # z = (B^2/u + B)/A <= MAX_BOUND => B^2/u <= A*MAX_BOUND - B => u >= B^2 / (A*MAX_BOUND - B)
            
            max_u_val = A * MAX_BOUND - B
            if max_u_val < 1:
                continue
                
            min_u_val = 1
            denom_z = A * MAX_BOUND - B
            if denom_z > 0:
                min_u_val = (B_sq + denom_z - 1) // denom_z # Ceiling division
            else:
                # If denom_z <= 0, z bound is always satisfied for u>=1?
                # No, if A*MAX_BOUND - B <= 0, then z > MAX_BOUND for all u?
                # z = (B^2/u + B)/A. If A*MAX_BOUND <= B, then B/A >= MAX_BOUND.
                # Then z >= B/A >= MAX_BOUND.
                # So no solution.
                continue

            divisors = get_sorted_divisors_sq(B)
            
            # Binary search for range [min_u_val, max_u_val]
            i_start = bisect.bisect_left(divisors, min_u_val)
            i_end = bisect.bisect_right(divisors, max_u_val)
            
            # Iterate divisors in range
            for i in range(i_start, i_end):
                u = divisors[i]
                if (u + B) % A != 0:
                    continue
                v = B_sq // u
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                # Bounds are already checked by u range, but verify distinctness
                if x == y or x == z or y == z:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
        
    return answers
