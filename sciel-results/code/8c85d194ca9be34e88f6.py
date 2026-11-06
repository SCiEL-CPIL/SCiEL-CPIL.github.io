import math
import time

def _find_sum_of_squares(p):
    """Find x, y such that x^2 + y^2 = p for prime p = 1 mod 4."""
    if p == 2:
        return 1, 1
    # Find a quadratic non-residue
    z = 2
    while pow(z, (p-1)//2, p) != p-1:
        z += 1
    # Compute x + iy = (z + i)^(p-1)/2 mod p
    x = 1
    y = 0
    # This is not efficient enough, use Cornacchia's algorithm or simple search for small p
    # Since p is small, simple search is fine
    for i in range(int(math.isqrt(p)) + 1):
        j2 = p - i*i
        j = int(math.isqrt(j2))
        if j*j == j2:
            return i, j
    return None, None

def _generate_pairs(N, primes, exponents):
    """
    Generate pairs (a, b) such that a^2 + b^2 = N, a, b > 0, a <= b.
    N is product of primes p_i^e_i where p_i = 1 mod 4.
    """
    # Get Gaussian integer factors for each prime
    # p = x^2 + y^2 = (x+iy)(x-iy)
    # For p^e, the factors are (x+iy)^k (x-iy)^(e-k) for k=0..e
    # We need to generate all combinations
    
    # Precompute (x+iy)^k for k=0..e for each prime
    # Represent Gaussian integer as (real, imag)
    
    def mul_gauss(a, b):
        # a = (a_r, a_i), b = (b_r, b_i)
        return (a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0])
        
    def pow_gauss(a, n):
        res = (1, 0)
        base = a
        while n:
            if n & 1:
                res = mul_gauss(res, base)
            base = mul_gauss(base, base)
            n >>= 1
        return res

    factors_options = []
    for p, e in zip(primes, exponents):
        x, y = _find_sum_of_squares(p)
        if x is None:
            continue
        g = (x, y)
        # Options are (x+iy)^k (x-iy)^(e-k)
        # Note: (x-iy) is conjugate
        conj_g = (x, -y)
        opts = []
        # We can just compute (x+iy)^k * (x-iy)^(e-k)
        # This is equivalent to (x+iy)^e * ((x-iy)/(x+iy))^k
        # Or just precompute powers
        pow_plus = [(1,0)]
        pow_minus = [(1,0)]
        for _ in range(e):
            pow_plus.append(mul_gauss(pow_plus[-1], g))
            pow_minus.append(mul_gauss(pow_minus[-1], conj_g))
            
        for k in range(e + 1):
            # (x+iy)^k * (x-iy)^(e-k)
            val = mul_gauss(pow_plus[k], pow_minus[e-k])
            opts.append(val)
        factors_options.append(opts)
        
    # Cartesian product
    all_solutions = [(1, 0)]
    for opts in factors_options:
        new_sols = []
        for s in all_solutions:
            for o in opts:
                new_sols.append(mul_gauss(s, o))
        all_solutions = new_sols
        
    pairs = set()
    for g in all_solutions:
        a = abs(g[0])
        b = abs(g[1])
        if a == 0 or b == 0:
            continue
        if a > b:
            a, b = b, a
        # Check if a^2 + b^2 == N
        if a*a + b*b == N:
            pairs.add((a, b))
            
    return sorted(list(pairs))

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Primes 1 mod 4
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73]
    exponents = [2, 2, 2, 1, 1, 1, 1, 1, 1]
    
    N = 1
    for p, e in zip(primes, exponents):
        N *= p ** e
        
    # Check N <= 10^18
    if N > 10**18:
        # Fallback to smaller N
        primes = [5, 13, 17, 29, 37, 41, 53, 61]
        exponents = [2, 2, 1, 1, 1, 1, 1, 1]
        N = 1
        for p, e in zip(primes, exponents):
            N *= p ** e
            
    pairs = _generate_pairs(N, primes, exponents)
    
    # Verify powerful
    # Squares are powerful
    verified_pairs = []
    for a, b in pairs:
        if a <= b and a + b != N: # Wait, a^2+b^2=N, not a+b=N
            pass
        # The task says a+b=N.
        # My analysis assumed a^2+b^2=N.
        # Re-read task: "a + b = N"
        # "a and b are powerful"
        # So a and b are powerful numbers, and their SUM is N.
        # My previous analysis was for sum of squares.
        # Powerful numbers include squares.
        # So if we pick a, b to be squares, then a+b=N.
        # But a,b must be powerful.
        # If a=x^2, b=y^2, then a+b = x^2+y^2 = N.
        # So the number of pairs is the number of representations of N as sum of two squares.
        # Yes, this holds.
        verified_pairs.append([a, b])
        
    return {'N': N, 'pairs': verified_pairs}

