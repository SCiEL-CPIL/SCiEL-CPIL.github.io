import time
import math

def factorize(num):
    factors = {}
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors[d] = factors.get(d, 0) + 1
            num //= d
        d += 1
    if num > 1:
        factors[num] = factors.get(num, 0) + 1
    return factors

def generate_divisors(factors):
    # factors is dict {p: e}
    # yields divisors of product(p^(e))
    # We want divisors of B^2, so exponents are 2*e
    primes = list(factors.keys())
    max_exps = [2 * factors[p] for p in primes]
    
    def gen(idx, current):
        if idx == len(primes):
            yield current
        else:
            p = primes[idx]
            for exp in range(max_exps[idx] + 1):
                yield from gen(idx + 1, current * (p ** exp))
    
    yield from gen(0, 1)

def construct():
    deadline = time.time() + 85.0
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # Try x from n//4 + 1 to n//4 + 100
        start_x = n // 4 + 1
        for x in range(start_x, start_x + 100):
            A = 4 * x - n
            B = n * x
            
            # X = A*y - B
            # Y = A*z - B
            # X*Y = B^2
            # X < B (to ensure y < z)
            # X > 4*x^2 - 2*n*x (to ensure y > x)
            lower_bound = 4 * x * x - 2 * n * x
            # If lower_bound < 1, set to 1
            if lower_bound < 1:
                lower_bound = 1
            
            # Factor B
            facs = factorize(B)
            
            # Generate divisors of B^2
            # We only need X <= B
            # And X > lower_bound
            # And X % A == (-B) % A
            # And (B^2 // X) % A == (-B) % A
            
            target_mod = (-B) % A
            
            for X in generate_divisors(facs):
                if X > B:
                    continue
                if X <= lower_bound:
                    continue
                if X % A != target_mod:
                    continue
                Y = B * B // X
                if Y % A != target_mod:
                    continue
                # Check y, z
                # y = (X+B)/A
                # z = (Y+B)/A
                # Ensure integers (guaranteed by mod check)
                # Ensure y > x (guaranteed by lower_bound)
                # Ensure z > y (guaranteed by X < Y, i.e., X < B)
                # If X == B, y == z, skip
                if X == B:
                    continue
                
                y = (X + B) // A
                z = (Y + B) // A
                
                # Final distinctness check
                if x != y and y != z and x != z:
                    # Check bounds
                    if x <= 10**12 and y <= 10**12 and z <= 10**12:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
