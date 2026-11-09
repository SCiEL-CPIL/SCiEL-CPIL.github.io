import time
def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    # Generate primes up to 1000 for trial division
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    # Add more primes up to 1000 if needed, but 100 is usually enough for LPF of numbers <= 10^6?
    # No, if n is prime, LPF is n.
    # But we are looking for MINIMUM LPF.
    # If n is prime, LPF is n.
    # We want small LPF.
    # So we only care about numbers that are NOT prime (or have small prime factors).
    # Actually, if n is prime, LPF(n) = n.
    # If n+1 is prime, LPF(n+1) = n+1.
    # We want to minimize max(LPF(n), LPF(n+1)).
    # So we want both to be composite with small factors.
    
    # To compute LPF(x), we can trial divide by primes.
    # If x becomes 1, LPF is the last prime used.
    # If x > 1 and no prime divides it, x is prime, LPF is x.
    
    best_val = float('inf')
    best_n = 10000
    
    for n in range(10000, 1000001):
        if time.time() > deadline:
            break
            
        lpf_n = get_lpf(n, primes)
        lpf_np1 = get_lpf(n+1, primes)
        
        val = max(lpf_n, lpf_np1)
        
        if val < best_val:
            best_val = val
            best_n = n
            
    return int(best_n)

def get_lpf(x, primes):
    orig = x
    lpf = 1
    for p in primes:
        if p * p > x:
            break
        while x % p == 0:
            lpf = p
            x //= p
    if x > 1:
        lpf = x
    return lpf
