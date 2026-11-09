"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
# EVOLVE-BLOCK-START
import time
import math

def _divisors(n):
    """Generate all divisors of n."""
    if n <= 0:
        return []
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(divs)

def construct():
    deadline = time.time() + 85  # safety margin before the 90s kill
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        # x is the smallest denominator. Since 1/x is the largest term,
        # 1/x < 4/n => x > n/4. Also 3/x >= 4/n => x <= 3n/4.
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            # a/x + b/y = 4/n - 1/x = (4x - n) / (nx)
            a = 4 * x - n          # positive since x > n/4
            b = n * x              # positive
            if a <= 0:
                continue
            
            # Simplify a/b by gcd
            g = math.gcd(a, b)
            a_red = a // g
            b_red = b // g
            
            # 1/y + 1/z = a_red / b_red
            # (a_red*y - b_red)(a_red*z - b_red) = b_red^2
            # Let d = a_red*y - b_red, then a_red*z - b_red = b_red^2 / d
            # y = (b_red + d) / a_red, z = (b_red + b_red^2/d) / a_red
            # Need d | b_red^2, and (b_red+d) % a_red == 0, and (b_red + b_red^2//d) % a_red == 0
            # Also y > x, z > y, all distinct, all <= 10^12
            
            b_red_sq = b_red * b_red
            divs = _divisors(b_red_sq)
            
            for d in divs:
                if time.time() > deadline:
                    break
                if found:
                    break
                
                # y = (b_red + d) / a_red
                num_y = b_red + d
                if num_y % a_red != 0:
                    continue
                y = num_y // a_red
                
                # z = (b_red + b_red^2/d) / a_red
                num_z = b_red + b_red_sq // d
                if num_z % a_red != 0:
                    continue
                z = num_z // a_red
                
                # Check ordering: x < y < z (or at least distinct)
                if y <= x or z <= y:
                    continue
                if x == y or y == z or x == z:
                    continue
                if y > 10**12 or z > 10**12 or x > 10**12:
                    continue
                
                # Verify via cross multiplication: 1/x+1/y+1/z == 4/n
                lhs_num = n * y * z + n * x * z + n * x * y
                rhs_num = 4 * x * y * z
                if lhs_num == rhs_num:
                    found = [x, y, z]
                    break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
