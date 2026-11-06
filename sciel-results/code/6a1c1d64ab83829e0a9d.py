import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    
    for n in range(3, 257):
        if time.time() > deadline:
            break
        idx = n - 3
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            b = n * x
            
            d = math.gcd(a, b)
            a_prime = a // d
            b_prime = b // d
            
            # Find divisors of b_prime
            divisors = []
            bp = b_prime
            isqrt_bp = math.isqrt(bp)
            for i in range(1, isqrt_bp + 1):
                if bp % i == 0:
                    divisors.append(i)
                    j = bp // i
                    if i != j:
                        divisors.append(j)
            
            for k in divisors:
                if (k + b_prime) % a_prime != 0:
                    continue
                y = (k + b_prime) // a_prime
                if y <= x:
                    continue
                z = b_prime * y // k
                if z <= y:
                    continue
                if x == y or y == z or x == z:
                    continue
                if x <= 10**12 and y <= 10**12 and z <= 10**12:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
    
    return answers
