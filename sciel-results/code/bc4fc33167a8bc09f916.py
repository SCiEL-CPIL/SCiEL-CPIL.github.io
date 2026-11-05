def check_smooth(n, p):
    if n == 1:
        return True
    # Trial division by small primes <= p
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for q in small_primes:
        if q > p:
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    if n > p and is_prime(n):
        return False
    # n is composite or prime <= p (if prime <= p, return True)
    if is_prime(n):
        return n <= p
    
    # n is composite.
    # Find a factor using Pollard Brent
    d = pollard_brent(n)
    if d == n:
        # Failure, try again or assume bad
        # For safety, return False if we can't factor?
        # But we need to be correct.
        # Let's retry with different c
        d = pollard_brent(n)
        if d == n:
            return False # Should not happen often
    
    if d > p:
        return False
    # d <= p. Check n//d
    return check_smooth(n // d, p)
