import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    answers = []
    
    # Precompute primes up to 1000 for factorization speed (though B is small)
    # Simple trial division is sufficient for B < 50000
    
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def get_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                val = 1
                for _ in range(e + 1):
                    new_divs.append(d * val)
                    val *= p
            divs = new_divs
        return divs

    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            B = n * x
            
            # U must divide B^2
            # U < B
            # U > 4x^2 - 2nx
            # (U+B) % A == 0
            
            factors = factorize(B)
            factors_sq = {p: 2*e for p, e in factors.items()}
            divisors = get_divisors(factors_sq)
            
            lower_bound = 4*x*x - 2*n*x
            upper_bound = B - 1
            
            for U in divisors:
                if U < lower_bound or U > upper_bound:
                    continue
                if (U + B) % A != 0:
                    continue
                
                y = (U + B) // A
                V = (B * B) // U
                z = (V + B) // A
                
                # Verify distinctness and order
                if x < y < z:
                    if z <= 10**12:
                        found = [x, y, z]
                        break
                        
            if found:
                break
                
        answers.append(found)
        
    return answers
