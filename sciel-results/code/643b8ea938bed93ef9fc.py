import math
import time

def construct():
    answers = [None] * 254
    start_time = time.time()
    
    # n ranges from 3 to 256
    for i, n in enumerate(range(3, 257)):
        # Check time
        if time.time() - start_time > 85:
            break
            
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4 (assuming x <= y <= z)
        # So x is in [floor(n/4)+1, floor(3n/4)]
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            A = 4 * x - n
            B = n * x
            
            # We need 1/y + 1/z = A/B
            # Reduce A/B
            g = math.gcd(A, B)
            A_red = A // g
            B_red = B // g
            
            # (A_red * y - B_red) must divide B_red * y
            # Let k = A_red * y - B_red. Then k must divide B_red.
            # Also y = (B_red + k) / A_red must be integer.
            # So A_red must divide B_red + k.
            # And z = B_red * y / k must be integer.
            
            # Iterate divisors k of B_red
            # k can be positive or negative?
            # Since y, z > 0, and A, B > 0, A_red * y > B_red, so k > 0.
            # So k is a positive divisor of B_red.
            
            # Find all divisors of B_red
            # B_red is at most n * 3n/4 / 1 = 3n^2/4. For n=256, B_red < 50000.
            # sqrt(50000) ~ 224. Very fast.
            
            # We can just iterate k from 1 to sqrt(B_red)
            limit = int(math.isqrt(B_red))
            
            for k in range(1, limit + 1):
                if B_red % k != 0:
                    continue
                
                # Check divisor k
                if (B_red + k) % A_red == 0:
                    y = (B_red + k) // A_red
                    # Check if y is integer (already ensured by mod check)
                    # Check if y > x
                    if y > x:
                        # Calculate z
                        # z = B_red * y / k
                        num_z = B_red * y
                        if num_z % k == 0:
                            z = num_z // k
                            if z > y:
                                # Check bounds
                                if z <= 1000000000000:
                                    found = [x, y, z]
                                    break
                
                # Check divisor B_red // k
                k2 = B_red // k
                if k2 == k:
                    continue
                if (B_red + k2) % A_red == 0:
                    y = (B_red + k2) // A_red
                    if y > x:
                        num_z = B_red * y
                        if num_z % k2 == 0:
                            z = num_z // k2
                            if z > y:
                                if z <= 1000000000000:
                                    found = [x, y, z]
                                    break
            
            if found:
                break
        
        answers[i] = found
        
    return answers
