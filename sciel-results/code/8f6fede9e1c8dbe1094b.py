"""Solve 1/x + 1/y + 1/z = 4/n for n=3..256 with distinct denominators <= 10^12."""
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    MAX_DEN = 10**12
    
    def get_divisors_of_square(b):
        """Generate all divisors of b^2."""
        # Factorize b
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
        
        # Generate divisors of b^2
        # If b = prod p_i^e_i, then b^2 = prod p_i^(2e_i)
        # Divisors are prod p_i^k_i where 0 <= k_i <= 2e_i
        divs = [1]
        for p, e in factors.items():
            current_divs = []
            power = 1
            for k in range(2 * e + 1):
                for d in divs:
                    current_divs.append(d * power)
                power *= p
            divs = current_divs
        return divs

    def find_solution(n):
        """Find distinct x<y<z with 1/x+1/y+1/z=4/n, denominators <= MAX_DEN."""
        # x must satisfy 1/x < 4/n => x > n/4
        # and x <= y <= z => 3/x >= 4/n => x <= 3n/4
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                return None
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need (ay - b)(az - b) = b^2
            # Let D = ay - b. D divides b^2.
            # y = (D + b) / a
            # z = (b^2/D + b) / a
            # Conditions:
            # 1. D | b^2
            # 2. (D + b) % a == 0
            # 3. (b^2/D + b) % a == 0
            # 4. y > x => D > ax - b
            # 5. y < z => D < b^2/D => D < b (since D>0)
            # 6. y, z <= MAX_DEN
            
            divs = get_divisors_of_square(b)
            
            for D in divs:
                if D >= b:
                    continue
                if D <= a * x - b:
                    continue
                
                num_y = D + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                
                num_z = b * b // D + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                
                if y <= x:
                    continue
                if y >= z:
                    continue
                if y > MAX_DEN or z > MAX_DEN:
                    continue
                
                return (x, y, z)
        
        return None
    
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        sol = find_solution(n)
        answers.append(sol)
    
    return answers
