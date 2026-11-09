import sys
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 1000000
    MAX_VAL = N + 1
    
    # 1. Sieve of Eratosthenes
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    
    for i in range(2, MAX_VAL + 1):
        if is_prime[i]:
            primes.append(i)
            if i * i <= MAX_VAL:
                for j in range(i * i, MAX_VAL + 1, i):
                    is_prime[j] = False
    
    # 2. Compute powerful part
    powerful_part = [1] * (MAX_VAL + 1)
    
    for p in primes:
        power = p * p
        while power <= MAX_VAL:
            next_power = power * p
            
            if next_power > MAX_VAL:
                for m in range(power, MAX_VAL + 1, power):
                    powerful_part[m] *= power
            else:
                for m in range(power, MAX_VAL + 1, power):
                    if m % next_power != 0:
                        powerful_part[m] *= power
            
            power = next_power
    
    # 3. Find max
    max_val = 0
    best_n = 10000
    
    for n in range(10000, N + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > max_val:
            max_val = val
            best_n = n
        
        if time.time() > deadline:
            break
            
    return best_n
