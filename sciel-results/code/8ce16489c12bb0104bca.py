import time

def construct():
    # The maximum size of a multiplicatively independent set from [2, 40]
    # is equal to the number of prime numbers in that range.
    # Primes in [2, 40]: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37.
    # There are 12 such primes.
    # These primes are linearly independent in the exponent vector space.
    # Thus, all subset products are distinct.
    # This is the theoretical maximum size.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify just in case (optional, but good for safety)
    # Check if all subset products are distinct
    # For 12 elements, 2^12 = 4096 products.
    # We can generate them to be absolutely sure.
    products = set()
    for i in range(1 << len(primes)):
        prod = 1
        for j in range(len(primes)):
            if (i >> j) & 1:
                prod *= primes[j]
        if prod in products:
            # Collision found, though theoretically impossible for primes
            pass
        products.add(prod)
        
    if len(products) == (1 << len(primes)):
        return primes
    else:
        # Fallback to a smaller known valid set if something went wrong
        return [2, 3, 5, 7, 11, 13]
