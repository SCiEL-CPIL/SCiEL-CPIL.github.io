"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    deadline = time.time() + 85
    answers = []
    BOUND = 10**12
    
    # Precompute primes up to sqrt(max B^2)
    # max n=256, max x ~ 192, max B = 256*192 = 49152
    # max B^2 ~ 2.4e9
    # sqrt(max B^2) = max B ~ 49152
    limit = 50000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, p in enumerate(is_prime) if p]
    
    def factorize(num):
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors

    def get_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        return divs

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Iterate x
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue
            
            # Factorize B^2
            # B = n*x. Since n is small, we can factorize B and double exponents for B^2
            # But factorizing B^2 directly is also fine as B is small.
            b_sq = B * B
            factors_b_sq = factorize(b_sq)
            divs = get_divisors(factors_b_sq)
            
            # We need u < v, so u < sqrt(uv) = B
            # Also y > x => (u+B)/A > x => u + B > Ax => u > Ax - B
            # Ax - B = x(4x-n) - nx = 4x^2 - nx - nx = 4x^2 - 2nx
            lower_bound_u = A * x - B
            
            for u in divs:
                if u >= B:
                    break
                if u <= lower_bound_u:
                    continue
                
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                if y <= x:
                    continue
                    
                v = b_sq // u
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                if z <= y:
                    continue
                
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                
                # Check bounds
                if z > BOUND:
                    continue
                    
                # Verify
                # 1/x + 1/y + 1/z = 4/n
                # (yz + xz + xy) / xyz = 4/n
                # n(yz + xz + xy) = 4xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
# EVOLVE-BLOCK-END
