import time
from math import gcd

def compute_tau_perp(n):
    """Compute the number of adjacent coprime pairs in sorted divisors of n."""
    if n < 2:
        return 0
    
    # Generate all divisors
    divisors = [1]
    temp = n
    i = 2
    while i * i <= temp:
        while temp % i == 0:
            current_len = len(divisors)
            for j in range(current_len):
                divisors.append(divisors[j] * i)
            temp //= i
        i += 1
    if temp > 1:
        current_len = len(divisors)
        for j in range(current_len):
            divisors.append(divisors[j] * temp)
    
    divisors.sort()
    
    # Count adjacent coprime pairs
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Start with squarefree product of first 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_exponents = [1] * 15
    best_n = 1
    for p, e in zip(primes, best_exponents):
        best_n *= p ** e
    best_score = compute_tau_perp(best_n)
    
    # Local search
    for iteration in range(50):
        if time.time() > deadline:
            return best_n
        
        improved = False
        for i in range(len(primes)):
            if time.time() > deadline:
                return best_n
            
            # Try increasing exponent
            for new_e in range(best_exponents[i] + 1, 15):
                new_n = best_n // (primes[i] ** best_exponents[i]) * (primes[i] ** new_e)
                if new_n > 10**18:
                    break
                
                score = compute_tau_perp(new_n)
                if score > best_score:
                    best_score = score
                    best_n = new_n
                    best_exponents[i] = new_e
                    improved = True
                    break
        
        if not improved:
            # Try decreasing exponents
            for i in range(len(primes) - 1, -1, -1):
                if time.time() > deadline:
                    return best_n
                if best_exponents[i] > 1:
                    new_e = best_exponents[i] - 1
                    new_n = best_n // (primes[i] ** best_exponents[i]) * (primes[i] ** new_e)
                    score = compute_tau_perp(new_n)
                    if score > best_score:
                        best_score = score
                        best_n = new_n
                        best_exponents[i] = new_e
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_n
