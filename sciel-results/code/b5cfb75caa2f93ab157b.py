import time

def construct():
    MAX_DENOM = 10**12
    deadline = time.time() + 85  # 85s deadline with 5s safety margin
    answers = [None] * 254  # for n=3..256
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        # x must satisfy: n/4 < x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            # We need (ay - b)(az - b) = b^2
            # u = ay - b, v = az - b, uv = b^2
            # y = (u + b) / a, z = (v + b) / a
            # Need u, v > 0, u <= v (so y <= z), u | b^2, v = b^2 // u
            # Need (u + b) % a == 0 and (v + b) % a == 0
            # Need y != x, z != x, y != z, all <= MAX_DENOM
            
            b_sq = b * b
            
            # Find divisors u of b_sq where u <= b (since u <= v and uv = b^2)
            # and u + b ≡ 0 (mod a)
            # u must be a divisor of b_sq
            # Iterate u from 1 to b, check if u | b_sq and (u + b) % a == 0
            
            # More efficient: iterate over divisors of b_sq that are <= b
            # But b_sq can be up to ~2.4e9, iterating to b is up to ~49152
            # That's fine
            
            for u in range(1, b + 1):
                if b_sq % u != 0:
                    continue
                if (u + b) % a != 0:
                    continue
                y = (u + b) // a
                v = b_sq // u
                if (v + b) % a != 0:
                    continue
                z = (v + b) // a
                # Check distinctness and bounds
                if y < x or z < y:
                    continue
                if y == x or z == x or y == z:
                    continue
                if y > MAX_DENOM or z > MAX_DENOM:
                    continue
                # Verify the solution
                # 1/x + 1/y + 1/z should equal 4/n
                # Cross multiply: n*(yz + xz + xy) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        
        answers[idx] = found
    
    return answers
