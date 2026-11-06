import math
import time

def _factorize(num):
    """Return prime factorization as dict {prime: exponent}."""
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

def _divisors_of_square(factors):
    """Given prime factorization of B, generate all divisors of B^2."""
    # B^2 has exponents 2*e for each prime with exponent e in B
    primes = list(factors.keys())
    max_exps = [2 * factors[p] for p in primes]
    
    def gen(idx, current):
        if idx == len(primes):
            yield current
            return
        p = primes[idx]
        for e in range(max_exps[idx] + 1):
            yield from gen(idx + 1, current * (p ** e))
    
    return list(gen(0, 1))

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy: n/4 < x <= 3n/4 (assuming x <= y <= z)
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need (A*y - B)(A*z - B) = B^2
            # u = A*y - B, v = A*z - B, u*v = B^2
            # y = (u + B) / A, z = (v + B) / A
            # Need y > x, z > y, all integers
            # u <= v means u <= B (since u*v = B^2)
            
            # Factor B to get divisors of B^2
            bfactors = _factorize(B)
            divs = _divisors_of_square(bfactors)
            
            for u in divs:
                if u > B:
                    continue
                # y = (u + B) / A
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                if y <= x:
                    continue
                # z = (B^2 // u + B) / A
                v = B * B // u  # since u divides B^2
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                if z <= y:
                    continue
                # Verify: all distinct, positive, within bounds
                if x < y < z and x > 0 and y > 0 and z > 0 and z <= 10**12:
                    # Double-check the equation
                    # 1/x + 1/y + 1/z = 4/n
                    # (yz + xz + xy) * n = 4 * xyz
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
