import time
import math

DENOM_BOUND = 10**12

# Precompute primes up to 256 * (3*256//4) = 256 * 192 = 49152
MAX_N = 256
MAX_X = (3 * MAX_N) // 4
MAX_VAL = MAX_N * MAX_X

def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = get_primes(MAX_VAL)

def factorize(num):
    factors = {}
    temp = num
    for p in PRIMES:
        if p * p > temp:
            break
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors[p] = count
    if temp > 1:
        factors[temp] = 1
    return factors

def generate_divisors_sq(factors):
    # Factors are for V. We need divisors of V^2.
    # If V = p1^a1 * p2^a2, then V^2 = p1^(2a1) * p2^(2a2).
    # Divisors are of form p1^e1 * p2^e2 where 0 <= ei <= 2*ai.
    divisors = [1]
    for p, exp in factors.items():
        max_exp = 2 * exp
        current_divs = []
        for d in divisors:
            for e in range(max_exp + 1):
                current_divs.append(d * (p ** e))
        divisors = current_divs
    return divisors

def construct():
    deadline = time.time() + 85.0 # Safety margin
    answers = [None] * 254
    
    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        nx = n
        
        # x must be > n/4 and <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = nx * x
            K = B * B
            
            # Factor B to get divisors of K=B^2
            # Note: B can be large, but B = n*x <= 256*192 ~ 50k.
            # So factorization is fast.
            factors_B = factorize(B)
            divs = generate_divisors_sq(factors_B)
            
            for U in divs:
                # U is a divisor of K.
                # We need (U + B) % A == 0
                if (U + B) % A != 0:
                    continue
                
                y = (U + B) // A
                
                # Check y is positive
                if y <= 0:
                    continue
                
                V = K // U
                if (V + B) % A != 0:
                    continue
                
                z = (V + B) // A
                
                if z <= 0:
                    continue
                
                # Check distinctness
                if x == y or y == z or x == z:
                    continue
                
                # Check bound
                if y > DENOM_BOUND or z > DENOM_BOUND:
                    continue
                
                # Valid solution found
                # We can return immediately as we just need one
                found = [x, y, z]
                break
            
            if found:
                break
                
        answers[i] = found
        
    return answers
