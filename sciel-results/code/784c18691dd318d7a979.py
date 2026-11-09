def construct():
    """
    Returns an integer n <= 10^18 that maximizes the number of adjacent coprime divisors.
    
    Analysis:
    1. The optimal number n is square-free.
    2. For a square-free number with k prime factors, the number of divisors is 2^k.
    3. To maximize the score, we should maximize k, the number of distinct prime factors.
    4. The product of the first 15 primes is approx 6.1 * 10^17, which is <= 10^18.
    5. The product of the first 16 primes is approx 3.2 * 10^19, which is > 10^18.
    6. Thus, the maximum number of prime factors is 15.
    7. Among all square-free numbers with 15 prime factors, the product of the 15 smallest primes (the primorial) yields the highest score for the number of adjacent coprime divisor pairs. This is because smaller primes lead to a more "interleaved" structure of divisors in terms of coprimality in the sorted order.
    
    Therefore, the optimal n is the product of the first 15 primes.
    """
    # The first 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    n = 1
    for p in primes:
        n *= p
        
    # n is 614889782588491410, which is < 10^18
    return n
