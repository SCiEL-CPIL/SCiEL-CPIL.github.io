import time
import math

def _factorize(num):
    factors = {}
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors[d] = factors.get(d, 0) + 1
            num //= d
        d += 1 if d == 2 else 2
    if num > 1:
        factors[num] = factors.get(num, 0) + 1
    return factors

def _get_divisors(factors):
    divisors = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divisors:
            pe = 1
            for _ in range(e + 1):
                new_divs.append(d * pe)
                pe *= p
        divisors = new_divs
    return divisors

def _find_solution(n):
    """Find distinct x,y,z with 1/x+1/y+1/z=4/n"""
    x_start = n // 4 + 1
    x_end = min(3 * n // 4, x_start + 50)
    
    for x in range(x_start, x_end + 1):
        K = 4 * x - n
        if K <= 0:
            continue
        nx = n * x
        N = nx * nx
        target_mod = (-nx) % K
        
        factors_nx = _factorize(nx)
        factors_N = {}
        for p, e in factors_nx.items():
            factors_N[p] = e * 2
        
        divisors = _get_divisors(factors_N)
        
        for U in divisors:
            if U % K != target_mod:
                continue
            V = N // U
            if V % K != target_mod:
                continue
            num_y = U + nx
            num_z = V + nx
            if num_y % K != 0 or num_z % K != 0:
                continue
            y = num_y // K
            z = num_z // K
            if y <= 0 or z <= 0:
                continue
            if x == y or x == z or y == z:
                continue
            if y <= 10**12 and z <= 10**12:
                return [x, y, z]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            while len(answers) < 254:
                answers.append(None)
            return answers
        
        found = _find_solution(n)
        answers.append(found)
    
    return answers
