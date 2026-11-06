import time

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457
    
    # Sieve primes up to TARGET
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(TARGET)
    prime_set = set(primes)
    
    # Generate all smooth numbers up to TARGET for a given bound B
    # A number is B-smooth if all its prime factors are <= B
    def generate_smooth_numbers(B, limit):
        small_primes = [p for p in primes if p <= B]
        if not small_primes:
            return set()
        
        smooth = {1}
        queue = [1]
        idx = 0
        while idx < len(queue):
            x = queue[idx]
            idx += 1
            for p in small_primes:
                y = x * p
                if y > limit:
                    break
                if y not in smooth:
                    smooth.add(y)
                    queue.append(y)
        return smooth
    
    # Iterate B from 2 upwards
    # For each B, generate all B-smooth numbers <= TARGET
    # Check if any two sum to TARGET
    for B in range(2, TARGET + 1):
        if time.time() > deadline - 5.0:
            # Fallback: use the simple approach
            # Just return a reasonable pair
            # We need to find any a, b with a+b=TARGET
            # Let's use a = 1, b = TARGET-1
            return [1, TARGET - 1]
        
        smooth = generate_smooth_numbers(B, TARGET)
        
        # Check if any two numbers in smooth sum to TARGET
        for a in smooth:
            if a > TARGET // 2:
                break
            b = TARGET - a
            if b in smooth:
                return [a, b]
    
    # Fallback (should not reach here)
    return [1, TARGET - 1]
