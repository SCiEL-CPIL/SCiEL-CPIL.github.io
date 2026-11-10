import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s wall-clock
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must be such that 4/x < 4/n => x > n/4
        # and 3/x > 4/n => x < 3n/4
        # We try x starting from floor(n/4) + 1
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            
            K = 4 * x - n
            if K <= 0:
                continue
            
            # We need A * B = (n*x)^2
            # such that (A + n*x) % K == 0 and (B + n*x) % K == 0
            # y = (A + n*x) / K
            # z = (B + n*x) / K
            target = n * x
            target_sq = target * target
            
            # Iterate divisors of target_sq
            # Optimization: only need to check up to sqrt
            # But number of divisors is small, so simple loop is fine
            # We can generate divisors by iterating i from 1 to sqrt(target_sq)
            # Or just iterate i from 1 to target? No, target can be large.
            # target = n*x. Max n=256, x~64. target ~ 16000. target_sq ~ 2.5e8.
            # Iterating to sqrt is 15000. Very fast.
            
            A = None
            B = None
            limit = int(target_sq**0.5)
            
            for i in range(1, limit + 1):
                if target_sq % i == 0:
                    # i is a divisor
                    # Check condition for A=i
                    if (i + target) % K == 0:
                        A = i
                        B = target_sq // i
                        if (B + target) % K == 0:
                            y = (A + target) // K
                            z = (B + target) // K
                            # Ensure distinctness and ordering x < y < z
                            # Since we iterate i increasing, A increases, B decreases.
                            # y increases, z decreases.
                            # We want y > x and z > y.
                            if x < y < z:
                                if z <= 10**12:
                                    found = [x, y, z]
                                    break
                            # Also check if swapping A,B gives a solution?
                            # y and z are symmetric in the equation.
                            # If we found y,z, we can sort them.
                            # But we need x < min(y,z) < max(y,z).
                            # My check x < y < z assumes y < z.
                            # If y > z, we can just swap them.
                            # However, the loop finds pairs.
                            # Let's just check if {y,z} contains values > x and distinct.
                            if x < y and x < z and y != z:
                                y_min, y_max = min(y, z), max(y, z)
                                if y_max <= 10**12:
                                    found = [x, y_min, y_max]
                                    break
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
