"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def construct():
    # The set of integers is constructed to be 1-independent (distinct subset products).
    # We select a maximal set of prime powers for each prime base p <= 40.
    # For p=2, powers are 2, 4, 8, 16, 32. Max 1-independent subset of exponents {1,2,3,4,5} containing 1 is {1,2,4} (values 2,4,16).
    # For p=3, powers are 3, 9, 27. Max 1-independent subset of exponents {1,2,3} containing 1 is {1,2} (values 3,9).
    # For p=5, powers are 5, 25. Max 1-independent subset of exponents {1,2} containing 1 is {1,2} (values 5,25).
    # For p>=7, only p itself is in range (p^2 > 40).
    # These groups are independent of each other.
    
    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Additional prime powers
    # Base 2: 4 (2^2), 16 (2^4). (2 is already in primes)
    # Base 3: 9 (3^2). (3 is already in primes)
    # Base 5: 25 (5^2). (5 is already in primes)
    
    selected = [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]
    
    # Verify types
    result = [int(x) for x in selected]
    return result
# EVOLVE-BLOCK-END
