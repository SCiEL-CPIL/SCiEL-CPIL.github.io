import math

def construct():
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    def phi_sieve(N):
        phi = list(range(N + 1))
        for i in range(2, N + 1):
            if phi[i] == i:
                for j in range(i, N + 1, i):
                    phi[j] -= phi[j] // i
        return phi

    def phi_fast(n, primes):
        if n == 1:
            return 1
        result = n
        y = n
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                while y % p == 0:
                    y //= p
                result -= result // p
        if y > 1:
            result -= result // y
        return result

    PRIME_LIMIT = 10**6
    primes = sieve_primes(PRIME_LIMIT)
    
    best_start = 1
    best_length = 1
    
    SIEVE_LIMIT = 10**7
    phi_vals = phi_sieve(SIEVE_LIMIT)
    
    seen = set()
    current_len = 0
    current_start = 1
    for i in range(1, SIEVE_LIMIT + 1):
        if phi_vals[i] not in seen:
            if current_len == 0:
                current_start = i
            current_len += 1
            seen.add(phi_vals[i])
            if current_len > best_length:
                best_length = current_len
                best_start = current_start
        else:
            current_len = 0
            seen = set()
            
    WINDOW_SIZE = 10**6
    start_points = [
        10**7, 2*10**7, 5*10**7,
        10**8, 2*10**8, 5*10**8,
        10**9, 2*10**9, 5*10**9,
        10**10, 10**11
    ]
    
    for start in start_points:
        end = start + WINDOW_SIZE
        if end > 10**12:
            end = 10**12
        if start >= end:
            continue
            
        seen = set()
        current_len = 0
        current_start = start
        
        for i in range(start, end + 1):
            p_val = phi_fast(i, primes)
            if p_val not in seen:
                if current_len == 0:
                    current_start = i
                current_len += 1
                seen.add(p_val)
                if current_len > best_length:
                    best_length = current_len
                    best_start = current_start
            else:
                current_len = 0
                seen = set()
                
        if best_length >= 200:
            break
            
    return {'start': best_start, 'length': best_length}
