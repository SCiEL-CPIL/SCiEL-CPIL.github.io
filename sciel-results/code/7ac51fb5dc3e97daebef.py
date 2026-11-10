import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    LIMIT = 10**12
    answers = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # x must satisfy 1/x < 4/n, so x > n/4
        # Also 1/x + 1/y + 1/z = 4/n with x <= y <= z implies 3/x >= 4/n, so x <= 3n/4
        # But we need distinct, so x < y < z
        # x_min = floor(n/4) + 1
        # x_max: since 1/x + 1/y + 1/z <= 3/x (since x<=y<=z), we need 3/x >= 4/n => x <= 3n/4
        # But actually 1/x + 1/y + 1/z = 4/n, and since y > x, z > y, we have 1/y < 1/x, 1/z < 1/y
        # So 4/n = 1/x + 1/y + 1/z < 1/x + 1/x + 1/x = 3/x => x < 3n/4
        # Also 4/n > 1/x => x > n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4  # x < 3n/4, so x_max = floor(3n/4) - 1 potentially, but let's use floor
        
        # For each x, we need 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
        # Let a = 4x - n, b = nx
        # 1/y + 1/z = a/b
        # (y+z)/(yz) = a/b
        # b(y+z) = a*yz
        # a*yz - b*y - b*z = 0
        # Multiply by a: a^2*yz - ab*y - ab*z = 0
        # (ay - b)(az - b) = b^2
        # Let u = ay - b, v = az - b, then uv = b^2
        # y = (u+b)/a, z = (v+b)/a
        # Need u,v positive divisors of b^2, u < v (since y < z)
        # Also y > x means (u+b)/a > x => u + b > ax => u > ax - b = a(4x-n) - nx = 4ax - an - nx
        # Wait, let me recompute: ax - b = a*x - b = (4x-n)*x - nx = 4x^2 - nx - nx = 4x^2 - 2nx
        # Hmm, let me just iterate over divisors
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # Find divisors of b^2 that give valid y, z
            # uv = b^2, y = (u+b)/a, z = (v+b)/a
            # y must be integer: (u+b) % a == 0
            # z must be integer: (v+b) % a == 0
            # y > x: (u+b) > a*x => u > a*x - b
            # z > y: v > u
            # y, z <= LIMIT
            
            # Instead of factoring b^2 (which can be huge), iterate y in a reasonable range
            # 1/y + 1/z = a/b
            # 1/z = a/b - 1/y = (ay - b)/(by)
            # z = by/(ay - b)
            # Need ay - b > 0 => y > b/a = nx/(4x-n)
            # z > y => by/(ay-b) > y => b/(ay-b) > 1 => b > ay - b => 2b > ay => y < 2b/a
            # So y ranges from floor(b/a) + 1 to floor(2b/a) - 1
            # Also y > x for distinctness
            
            y_min = max(x + 1, b // a + 1)
            y_max = 2 * b // a - 1
            
            if y_min > y_max:
                continue
            
            # Iterate y and check if z is integer
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                denom = a * y - b
                if denom <= 0:
                    continue
                if (b * y) % denom != 0:
                    continue
                z = (b * y) // denom
                if z <= y:
                    continue
                if z > LIMIT:
                    continue
                # Verify
                # 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n(yz + xz + xy) == 4xyz
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
