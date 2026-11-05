import math

def factorize(n):
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors

def get_divisors_of_square(B):
    # B is an integer. We want divisors of B^2.
    # Factor B first.
    factors = factorize(B)
    # The prime factorization of B^2 is p_i^(2*e_i)
    # Divisors are p_1^k1 ... pk^kk where 0 <= ki <= 2*ei
    
    divisors = [1]
    for p, e in factors:
        new_divs = []
        max_exp = 2 * e
        for d in divisors:
            for k in range(max_exp + 1):
                new_divs.append(d * (p ** k))
        divisors = new_divs
    return divisors

def construct():
    answers = []
    for n in range(3, 257):
        found = None
        start_x = n // 4 + 1
        # Limit x search range to keep it fast, but ensure we don't miss solutions.
        # Usually small x works.
        for x in range(start_x, start_x + 100):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need U * V = B^2
            # U % A == (-B) % A
            # V % A == (-B) % A
            target_rem = (-B) % A
            
            divs = get_divisors_of_square(B)
            
            # Optimization: check if any divisor matches the remainder
            # We can filter divs first?
            # There might be many divisors.
            # Let's iterate.
            for U in divs:
                if U % A == target_rem:
                    V = (B * B) // U
                    if V % A == target_rem:
                        y = (U + B) // A
                        z = (V + B) // A
                        if x != y and x != z and y != z:
                            if x <= 1000000000000 and y <= 1000000000000 and z <= 1000000000000:
                                found = [x, y, z]
                                break
            if found:
                break
        answers.append(found)
    return answers
