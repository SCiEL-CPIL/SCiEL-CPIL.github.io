import time

DENOM_BOUND = 10**12

def _factorize(num):
    """Factorize a number into prime factors."""
    if num <= 1:
        return {}
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

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        pe = 1
        for i in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _solve_for_x(n, x, deadline):
    """Given n and x, find y and z such that 1/x + 1/y + 1/z = 4/n."""
    A = 4 * x - n
    if A <= 0:
        return None
    B = n * x
    
    # u = A*y - B, so y = (u + B) / A
    # v = B^2 / u, so z = (v + B) / A
    # Need: u*v = B^2, u > 0, v > 0
    # y > x => (u+B)/A > x => u > A*x - B
    # z > y => v > u
    # z <= DENOM_BOUND => (v+B)/A <= DENOM_BOUND => v <= A*DENOM_BOUND - B
    
    u_min = A * x - B + 1
    u_max = B - 1  # since v = B^2/u > u requires u < B
    
    if u_min > u_max:
        return None
    
    B_sq = B * B
    
    # Try divisor approach first
    B_factors = _factorize(B)
    B_sq_factors = {p: 2 * e for p, e in B_factors.items()}
    B_sq_divs = _divisors_from_factors(B_sq_factors)
    
    for u in B_sq_divs:
        if time.time() > deadline:
            return None
        if u < u_min or u > u_max:
            continue
        v = B_sq // u
        if v <= u:
            continue
        if (u + B) % A != 0:
            continue
        if (v + B) % A != 0:
            continue
        y = (u + B) // A
        z = (v + B) // A
        if y <= x or z <= y:
            continue
        if z > DENOM_BOUND:
            continue
        if x == y or y == z or x == z:
            continue
        return [x, y, z]
    
    return None

def construct():
    deadline = time.time() + 85.0
    results = [None] * 254
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        t_start = time.time()
        remaining = deadline - t_start
        # Budget per n: at least 0.1s, at most 0.3s
        n_budget = max(0.1, min(0.3, remaining / max(1, 256 - idx)))
        
        x_min = n // 4 + 1
        
        # Try x values starting from x_min
        # Limit to reasonable range based on budget
        max_x_tries = 200
        for x in range(x_min, x_min + max_x_tries):
            if time.time() > deadline or (time.time() - t_start) > n_budget:
                break
            
            result = _solve_for_x(n, x, deadline)
            if result is not None:
                found = result
                break
        
        results[idx] = found
    
    return results
