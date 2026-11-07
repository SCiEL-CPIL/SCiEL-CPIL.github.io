import math
import time

def construct():
    n = 256
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Sieve of Eratosthenes to find primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    result = [1]
    
    for p in primes:
        power = p
        while power * p <= n:
            power *= p
        result.append(power)
    
    # Local search: try 2-exchange moves to improve the sum
    # For each pair (a, b) in result, try replacing a with a' > a where gcd(a', b) = 1
    # and gcd(a', c) = 1 for all c != a, b in result
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        current_sum = sum(result)
        for i in range(len(result)):
            if time.time() > deadline:
                break
            a = result[i]
            # Try to replace a with a larger number
            for a_new in range(a + 1, n + 1):
                if time.time() > deadline:
                    break
                # Check if a_new is coprime to all other elements
                ok = True
                for j in range(len(result)):
                    if j == i:
                        continue
                    if math.gcd(a_new, result[j]) != 1:
                        ok = False
                        break
                if ok:
                    # Also need to check that no two elements are equal
                    if a_new not in result:
                        old_sum = current_sum
                        current_sum = old_sum - a + a_new
                        result[i] = a_new
                        improved = True
                        if current_sum > old_sum:
                            break
    return result
