import time

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin
    
    # Precompute primes up to 500
    limit = 500
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def factorize(num):
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors

    def get_divisors_from_factors(factors):
        divisors = [1]
        for p, e in factors.items():
            current_divs = []
            for d in divisors:
                power = 1
                for _ in range(e + 1):
                    current_divs.append(d * power)
                    power *= p
            divisors = current_divs
        return divisors

    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            while len(answers) < 254:
                answers.append(None)
            return answers
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            B = n * x
            
            b_factors = factorize(B)
            b_sq_factors = {}
            for p, e in b_factors.items():
                b_sq_factors[p] = 2 * e
                
            divisors = get_divisors_from_factors(b_sq_factors)
            
            for D in divisors:
                if D >= B:
                    continue
                if (D + B) % A != 0:
                    continue
                    
                y = (D + B) // A
                if y <= x:
                    continue
                    
                num_z = (B * B) // D + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                if z > y:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
