import math
import time

def construct():
    deadline = time.time() + 75  # 75s safety margin (90s limit)
    answers = [None] * 254  # for n=3..256
    
    # Precompute: for each n, find solutions
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = _find_solution(n)
        answers[idx] = found
    
    return answers

def _find_solution(n):
    """Find distinct positive integers x,y,z with 1/x+1/y+1/z=4/n, x<y<z, z<=10^12."""
    # We need 1/x + 1/y + 1/z = 4/n
    # => (4n - yz)/(xyz) = 4/n... let me use the standard approach
    # 1/x + 1/y + 1/z = 4/n
    # Multiply by xyz: yz + xz + xy = 4xyz/n
    # So 4xyz - n(xy + xz + yz) = 0
    # Rearrange: (4x - n)(4y - n)(4z - n) = n^3 + ... let me use a different substitution
    
    # Standard: let a=4x-n, b=4y-n, c=4z-n
    # Then x=(a+n)/4, y=(b+n)/4, z=(c+n)/4
    # 1/x + 1/y + 1/z = 4/n
    # 4/(a+n) + 4/(b+n) + 4/(c+n) = 4/n
    # 1/(a+n) + 1/(b+n) + 1/(c+n) = 1/n
    
    # Alternative approach: fix x, solve for y,z
    # 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
    # So (y+z)/(yz) = (4x-n)/(nx)
    # nx(y+z) = (4x-n)yz
    # Let A = 4x - n, B = nx
    # Then B(y+z) = A*yz
    # A*yz - B*y - B*z = 0
    # (Ay - B)(Az - B) = B^2
    # Let u = Ay - B, v = Az - B, then u*v = B^2
    # y = (u+B)/A, z = (v+B)/A
    # Need: u,v positive divisors of B^2, u <= v (so y <= z)
    # y > x means (u+B)/A > x => u+B > Ax => u > Ax - B = x(4x-n) - nx = 4x^2 - nx - nx = 4x^2 - 2nx
    # Wait, let me redo: Ax = x(4x-n), so u > x(4x-n) - nx = 4x^2 - nx - nx = 4x^2 - 2nx
    # Also need y,z integers: A divides (u+B) and A divides (v+B)
    # y, z distinct from x and from each other
    
    # x must satisfy: 4x - n > 0 => x > n/4
    # Also 1/x < 4/n (since 1/y + 1/z > 0) => x > n/4
    # And 1/x > 4/n - 2/z_min... roughly x < n/4 * something
    # Actually x can be at most... if y=x+1, z large: 1/x + 1/(x+1) + epsilon = 4/n
    # So x is bounded. Let's just iterate x from floor(n/4)+1 upward.
    
    # For efficiency, we iterate x and for each x, enumerate divisors of B^2 = (nx)^2
    # B = nx, so B^2 = n^2 * x^2
    # Divisors of n^2*x^2: we can enumerate by factoring n and x
    
    # But factoring for each x might be slow. Alternative: iterate u over divisors of B^2.
    # B^2 can be large. For n up to 256 and x up to maybe a few hundred, B^2 is manageable.
    
    # Let's try: for x in range(n//4 + 1, ...):
    #   A = 4*x - n
    #   B = n*x
    #   B2 = B*B
    #   For each divisor u of B2 with u <= sqrt(B2):
    #     v = B2 // u
    #     Check if (u+B) % A == 0 and (v+B) % A == 0
    #     y = (u+B)//A, z = (v+B)//A
    #     Check y > x, z > y, all distinct, z <= 10^12
    
    # To enumerate divisors of B2 efficiently, factor B2.
    # B = n*x, so B2 = n^2 * x^2
    # Factor n and x separately, combine.
    
    # Actually, for small n (3..256), n is small. Let me just trial-divisor up to sqrt(B2).
    # sqrt(B2) = B = n*x. For n=256, x could be up to ~200, so B~51200, sqrt~51200. That's fine.
    
    # But we want to find ANY solution quickly. Let's try small x first.
    
    max_x = n // 4 + 500  # generous bound
    for x in range(n // 4 + 1, max_x + 1):
        A = 4 * x - n
        if A <= 0:
            continue
        B = n * x
        B2 = B * B
        
        # Enumerate divisors u of B2 where u <= B (so u <= v)
        # u must satisfy: u > x*A - B (for y > x)
        # y = (u+B)/A > x => u > x*A - B
        min_u = x * A - B + 1  # u must be at least this
        if min_u < 1:
            min_u = 1
        
        # Iterate u from min_u to sqrt(B2)
        # But we need u to divide B2. Let's iterate all u in range and check divisibility.
        # This could be slow. Better: enumerate divisors.
        
        # For speed, let's iterate u from max(1, min_u) to int(sqrt(B2))+1
        # and check if B2 % u == 0
        upper = int(math.isqrt(B2))
        for u in range(max(1, min_u), upper + 1):
            if B2 % u != 0:
                continue
            v = B2 // u
            # Check divisibility
            if (u + B) % A != 0:
                continue
            if (v + B) % A != 0:
                continue
            y = (u + B) // A
            z = (v + B) // A
            # Check conditions: x < y < z, all distinct, positive, z <= 10^12
            if y <= x or z <= y:
                continue
            if z > 10**12:
                continue
            if x == y or y == z or x == z:
                continue
            return [x, y, z]
    
    return None
