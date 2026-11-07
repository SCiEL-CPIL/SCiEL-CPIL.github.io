import time

def construct():
    # The problem asks for a set of integers in 2..40 such that all subset products are distinct.
    # This is equivalent to finding a dissociated set of prime factorization vectors.
    # 
    # Analysis of maximum size:
    # 1. Primes <= 40 are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 (12 primes).
    #    Including all 12 primes is optimal as they provide independent dimensions.
    # 2. We can add powers of these primes.
    #    For a prime p, if we include $p^{e_1}, \dots, p^{e_k}$, the exponents must form a dissociated set.
    #    For p=2, powers <= 40 are 2, 4, 8, 16, 32 (exponents 1, 2, 3, 4, 5).
    #    Max dissociated subset size is 3 (e.g., 1, 2, 4). So we can add 2 extra powers (4, 16) to the prime 2.
    #    For p=3, powers <= 40 are 3, 9, 27 (exponents 1, 2, 3).
    #    Max dissociated subset size is 2 (e.g., 1, 2). So we can add 1 extra power (9) to the prime 3.
    #    For p=5, powers <= 40 are 5, 25 (exponents 1, 2).
    #    Max dissociated subset size is 2. So we can add 1 extra power (25) to the prime 5.
    #    For p >= 7, only the prime itself is <= 40. So 0 extra powers.
    # 
    # Total size = 12 (primes) + 2 (from 2) + 1 (from 3) + 1 (from 5) = 16.
    # 
    # This construction is guaranteed to be valid and is maximal.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Add specific powers
    # For 2: add 4 (2^2) and 16 (2^4). Exponents 1, 2, 4 are dissociated.
    # For 3: add 9 (3^2). Exponents 1, 2 are dissociated.
    # For 5: add 25 (5^2). Exponents 1, 2 are dissociated.
    
    result = primes[:]
    result.append(4)
    result.append(16)
    result.append(9)
    result.append(25)
    
    # Ensure all are in 2..40
    result = [x for x in result if 2 <= x <= 40]
    
    return result
