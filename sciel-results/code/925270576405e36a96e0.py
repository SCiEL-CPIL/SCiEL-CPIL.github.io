import random

def construct():
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    def small_prime_part(x):
        result = x
        for p in small_primes:
            if result % p == 0:
                while result % p == 0:
                    result //= p
        return x // result

    def score(n):
        factors = set()
        for i in range(1, 25):
            factors.add(small_prime_part(n + i))
        return len(factors)

    best_n = 1000
    best_count = 25
    
    # Coarse search: step size 100
    for n in range(1000, 1000001, 100):
        c = score(n)
        if c < best_count:
            best_count = c
            best_n = n

    # Local refinement around the best candidate found in coarse search
    start = max(1000, best_n - 200)
    end = min(1000000, best_n + 200)
    
    for n in range(start, end + 1):
        c = score(n)
        if c < best_count:
            best_count = c
            best_n = n
            if best_count == 1:
                break

    # If the coarse search didn't find anything great, do some random sampling
    if best_count > 10:
        for _ in range(5000):
            n = random.randint(1000, 1000000)
            c = score(n)
            if c < best_count:
                best_count = c
                best_n = n

    return best_n
