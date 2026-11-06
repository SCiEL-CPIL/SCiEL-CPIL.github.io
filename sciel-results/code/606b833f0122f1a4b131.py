import time
import math

def construct():
    deadline = time.time() + 78.0
    answers = [None] * 254
    BOUND = 10**12
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
        found = None
        
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z, all distinct positive integers
        # x must be > n/4, so x_min = floor(n/4) + 1
        # For a given x, we have 1/y + 1/z = (4x - n)/(nx)
        # Let a = 4x - n, b = nx
        # Then 1/y + 1/z = a/b, so (a*y - b)*z = b*y
        # z = b*y / (a*y - b)
        # For z to be integer, (a*y - b) must divide b*y
        # Note: gcd(a*y - b, b) = gcd(a*y, b) = gcd(a, b) * gcd(y, b/gcd(a,b))
        # Actually, let's use: z = b*y/(a*y - b)
        # Let d = gcd(a, b). Then a = d*a', b = d*b'
        # z = d*b'*y / (d*a'*y - d*b') = b'*y / (a'*y - b')
        # So we need (a'*y - b') | b'*y
        # gcd(a'*y - b', b') = gcd(a'*y, b') = gcd(y, b') since gcd(a',b')=1
        # So we need gcd(y, b') | b' which is always true
        # Wait, that's not quite right. Let me reconsider.
        
        # z = b*y / (a*y - b)
        # Let g = gcd(a*y - b, b)
        # We need g | b*y
        # Since g | b, we need g | b*y, which means g/gcd(g,b) | y, but g|b so gcd(g,b)=g
        # So we need 1 | y, which is always true? No.
        
        # Let me redo: z = b*y/(a*y-b)
        # Let h = gcd(a*y-b, b)
        # Then z is integer iff h | b*y
        # Since h | b, write h = h1 * h2 where h1 | b and h1 = gcd(h, b) = h (since h|b)
        # Actually h | b by definition. So h | b*y iff h | b*y.
        # Since h | b, h | b*y is equivalent to (b/h) * (y) being integer... no.
        # h | b*y means b*y = h*k for some integer k.
        # Since h | b, b = h*m, so h*m*y = h*k, meaning m*y = k, which is always integer.
        # So z is ALWAYS an integer when a*y - b divides b? No, that's not right either.
        
        # Let's just check: z = b*y / (a*y - b)
        # We need (a*y - b) | (b*y)
        # Note that gcd(a*y - b, b) = gcd(a*y, b) (since gcd(x-b, b) = gcd(x, b))
        # Let g = gcd(a*y, b) = gcd(a, b) * gcd(y, b/gcd(a,b))... not exactly.
        # Let d = gcd(a, b). Then gcd(a*y, b) = d * gcd(a*y/d, b/d) = d * gcd(a'*y, b')
        # where a = d*a', b = d*b', gcd(a', b') = 1.
        # gcd(a'*y, b') = gcd(y, b') since gcd(a', b') = 1.
        # So gcd(a*y - b, b) = d * gcd(y, b')
        # For z to be integer, we need d*gcd(y,b') | b*y = d*b'*y
        # This means gcd(y,b') | b'*y, which is always true.
        # Wait, so z is always an integer? Let me verify with an example.
        # n=3: x_min = 3//4+1 = 1. But x must be > n/4 = 0.75, so x>=1.
        # But x,y,z are distinct positive integers and 1/x+1/y+1/z = 4/3.
        # If x=1: 1/y+1/z = 4/3-1 = 1/3. So y,z are solution to 1/y+1/z=1/3.
        # y must be > 3, so y>=4. z = 3y/(y-3).
        # y=4: z=12/1=12. Check: 1/1+1/4+1/12 = 1+0.25+0.0833 = 1.333 = 4/3. Yes!
        
        # So the key insight: for given x, we need to find y such that (a*y-b) | b*y and z>y.
        # From the analysis, z = b*y/(a*y-b) is integer iff... let me just compute directly.
        
        # Actually, I realize the divisibility condition can be simplified.
        # z = b*y/(a*y-b). Let me set t = a*y - b, so y = (t+b)/a.
        # For y to be integer, a | (t+b).
        # z = b*(t+b)/(a*t)
        # This is getting complicated. Let's just iterate y and check.
        
        # For efficiency, note that y must satisfy:
        # 1/y < a/b (since 1/z > 0), so y > b/a
        # Also 1/y > a/b - 1/b = (a-1)/b... no.
        # 1/y + 1/z = a/b, and z > y, so 1/z < 1/y, thus 2/y > a/b, so y < 2b/a
        # So y is in range (b/a, 2b/a)
        
        x_min = n // 4 + 1
        # Try multiple x values for better coverage
        for x in range(x_min, min(x_min + 500, n * 2 + 10)):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # y must satisfy: b/a < y < 2*b/a
            # y must be > x (for distinctness and ordering)
            y_low = max(x + 1, b // a + 1)
            y_high = 2 * b // a
            
            if y_high < y_low:
                continue
            
            # Iterate y in [y_low, y_high]
            # To speed up, we can use the fact that (a*y - b) must divide b*y
            # Let d = gcd(a, b)
            d = math.gcd(a, b)
            a_prime = a // d
            b_prime = b // d
            
            # z = b*y/(a*y-b) = d*b_prime*y/(d*a_prime*y - d*b_prime) = b_prime*y/(a_prime*y - b_prime)
            # Need (a_prime*y - b_prime) | b_prime*y
            # gcd(a_prime*y - b_prime, b_prime) = gcd(a_prime*y, b_prime) = gcd(y, b_prime) [since gcd(a_prime,b_prime)=1]
            # So we need gcd(y, b_prime) | b_prime*y, which is always true.
            # Wait, that means z is always an integer? Let me double-check.
            
            # z = b_prime * y / (a_prime * y - b_prime)
            # Let h = gcd(a_prime*y - b_prime, b_prime)
            # h = gcd(a_prime*y, b_prime) = gcd(y, b_prime) since gcd(a_prime, b_prime) = 1
            # For z to be integer, we need h | b_prime * y
            # h | b_prime (by definition) and h | y (since h = gcd(y,b_prime))
            # So h | b_prime * y is always true.
            # Therefore z is ALWAYS an integer for any y in the valid range!
            
            # So we just need to find any y in [y_low, y_high] such that z > y.
            # z = b_prime * y / (a_prime * y - b_prime)
            # z > y iff b_prime * y > y * (a_prime * y - b_prime)
            # iff b_prime > a_prime * y - b_prime  [dividing by y > 0]
            # iff 2 * b_prime > a_prime * y
            # iff y < 2 * b_prime / a_prime
            # But we already have y < 2*b/a = 2*b_prime*d/(a_prime*d) = 2*b_prime/a_prime
            # So z > y is automatically satisfied for all y in (b/a, 2*b/a)!
            
            # Wait, that means the first valid y gives a solution? Let me verify.
            # n=3, x=1: a=1, b=3. y_low = max(2, 3//1+1) = 4. y_high = 2*3//1 = 6.
            # y=4: z = 3*4/(1*4-3) = 12/1 = 12. z>y. Solution: (1,4,12). Correct!
            # n=4, x_min=2. x=2: a=4, b=8. y_low=max(3,8//4+1)=max(3,3)=3. y_high=2*8//4=4.
            # y=3: z=8*3/(4*3-8)=24/4=6. Check: 1/2+1/3+1/6=0.5+0.333+0.1667=1.0=4/4. Correct!
            
            # So for each x, the smallest valid y gives a solution if y_high >= y_low.
            # But we need distinct x,y,z. x < y < z is guaranteed by construction.
            
            if y_high >= y_low:
                y = y_low
                z = b * y // (a * y - b)
                if z > y and z <= BOUND and y <= BOUND and x <= BOUND:
                    # Verify: 1/x + 1/y + 1/z == 4/n
                    # Cross-multiply: n*y*z + n*x*z + n*x*y == 4*x*y*z
                    lhs = n * y * z + n * x * z + n * x * y
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
        
        answers[idx] = found
    
    return answers
