    def is_p_smooth(n, p):
        # Trial divide by primes up to 10^5
        for q in primes:
            if q > 10**5: break
            if q * q > n: break
            while n % q == 0:
                n //= q
        if n == 1:
            return True
        if is_probable_prime(n):
            return n <= p
        # n is composite
        # Factor n
        factors = []
        def factorize(x):
            if x == 1: return
            if is_probable_prime(x):
                factors.append(x)
                return
            d = pollard_brent(x)
            factorize(d)
            factorize(x // d)
        factorize(n)
        for f in factors:
            if f > p:
                return False
        return True
