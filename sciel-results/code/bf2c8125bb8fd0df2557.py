import time

def construct():
    N = 1000000
    # Initialize phi[i] = i
    phi = list(range(N + 1))
    
    # Sieve for totient
    # Iterate through numbers to find primes
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] = phi[j] // i * (i - 1)
                
    # Find smallest n with phi[n] divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
            
    return -1
