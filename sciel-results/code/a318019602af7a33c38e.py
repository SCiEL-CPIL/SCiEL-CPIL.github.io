import time
import math
from math import gcd

def construct():
    deadline = time.time() + 82.0
    answers = [None] * 254
    MAX_DEN = 10**12

    def get_divisors_of_square(b):
        """
        Efficiently generate all positive divisors of b^2.
        b is at most ~50000, so b^2 is at most ~2.5e9.
        Factorization of b is fast. Generating divisors of b^2 is fast.
        """
        n = b
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        
        divs = [1]
        for p, e in factors.items():
            exp = 2 * e
            current_divs = []
            for d in divs:
                val = d
                for _ in range(exp + 1):
                    current_divs.append(val)
                    val *= p
            divs = current_divs
        return divs

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        
        found = None
        x_min = n // 4 + 1
        x_max = 3 * n // 4

        # Iterate x in increasing order.
        # Smaller x generally leads to larger y and z, but the search space for D is structured.
        # However, for distinctness x < y < z, we want to find a valid triple quickly.
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            
            b = n * x
            
            # We are solving for D such that:
            # (a*y - b) = D
            # (a*z - b) = b^2 / D
            # So D must be a divisor of b^2.
            # y = (D + b) / a
            # z = (b^2 / D + b) / a
            
            # Conditions:
            # 1. D | b^2
            # 2. (D + b) % a == 0  => y is integer
            # 3. (b^2 / D + b) % a == 0 => z is integer
            # 4. y > x  => (D + b) / a > x  => D + b > a*x  => D > a*x - b
            #    Note: a*x - b = (4x-n)x - nx = 4x^2 - nx - nx = 4x^2 - 2nx.
            #    Also D > 0.
            # 5. z > y  => (b^2/D + b) > (D + b)  => b^2/D > D  => b^2 > D^2  => D < b (since D>0)
            # 6. z <= MAX_DEN
            
            # Optimization:
            # Instead of generating ALL divisors of b^2, which can be many,
            # we can iterate through divisors and filter.
            # But for b up to 50k, number of divisors of b^2 is small (max ~1000-2000).
            # So generating them is acceptable.
            
            divs = get_divisors_of_square(b)
            
            # To find a solution quickly, we might want to check divisors in a specific order?
            # Usually, smaller D gives larger z. Larger D (closer to b) gives z closer to y.
            # We just need ANY distinct solution.
            
            # Sort divisors? Not strictly necessary, but might help if there's a bias.
            # Let's just iterate.
            
            for D in divs:
                if D >= b:
                    # z <= y or z == y (if D=b)
                    # If D=b, z=y. We need distinct.
                    # If D > b, z < y.
                    # Since we assume x < y < z, we need D < b.
                    continue
                
                # Check y > x
                # D > a*x - b
                # Note: a*x - b can be negative. If a*x - b < 0, D > negative is always true for D>0.
                lower_bound_D = a * x - b
                if D <= lower_bound_D:
                    continue
                
                # Check y integer
                if (D + b) % a != 0:
                    continue
                
                y = (D + b) // a
                
                # Check y > x again (redundant but safe)
                if y <= x:
                    continue
                
                # Check z integer
                term = b * b // D
                if (term + b) % a != 0:
                    continue
                
                z = (term + b) // a
                
                # Check z > y
                if z <= y:
                    continue
                
                # Check bound
                if z > MAX_DEN:
                    continue
                
                # Valid solution found
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found

    return answers
