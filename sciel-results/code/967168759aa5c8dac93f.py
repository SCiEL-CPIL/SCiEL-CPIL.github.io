def construct():
    """
    Returns an integer n <= 10^18 that maximizes the number of adjacent 
    coprime pairs in its sorted divisor list.
    
    The optimal strategy is to use the largest possible number of distinct 
    prime factors. For n <= 10^18, the maximum number of distinct prime 
    factors is 15 (the product of the first 15 primes is ~6.15 * 10^17, 
    while adding the 16th prime exceeds 10^18).
    
    A square-free number with k prime factors has a divisor lattice 
    isomorphic to a k-dimensional hypercube. This structure maximizes 
    the potential for adjacent divisors to have disjoint prime supports 
    compared to numbers with repeated prime factors, which cluster divisors 
    more closely in value and increase the likelihood of shared factors 
    among neighbors.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    return n
