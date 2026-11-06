"""Rewritten program: maximize coprime adjacent divisor pairs via primorial."""
# EVOLVE-BLOCK-START
def construct():
    # Use the product of the first 15 primes (primorial 47#)
    # This gives 2^15 = 32768 divisors, maximizing opportunities for coprime adjacencies
    # while staying under 10^18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    return n
# EVOLVE-BLOCK-END
