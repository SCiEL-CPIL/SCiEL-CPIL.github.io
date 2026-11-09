def construct():
    N = 256
    
    # Step 1: Find all primes up to N
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [p for p in range(2, N + 1) if is_prime(p)]
    
    # Step 2: For each prime, find the largest power <= N
    # This is optimal because any composite number x = p*q... uses primes p, q...
    # and is always less than the sum of the largest powers of p, q... individually
    # or at least, replacing a composite with its prime power components increases the sum
    # while maintaining coprimality (since prime powers of distinct primes are coprime).
    result = [1]  # 1 is coprime to all integers
    
    for p in primes:
        max_power = p
        current = p
        while current * p <= N:
            current *= p
            max_power = current
        result.append(max_power)
        
    # Ensure types are standard Python ints
    return [int(x) for x in result]
