    def is_p_smooth(n, p):
        temp_n = n
        for q in small_primes:
            if q * q > temp_n:
                break
            if temp_n % q == 0:
                if q > p:
                    return False
                while temp_n % q == 0:
                    temp_n //= q
        
        if temp_n == 1:
            return True
        
        if temp_n <= p:
            return True
        
        # temp_n > p
        if is_prime(temp_n):
            return False
        
        # temp_n is composite and > p
        # Factor it
        def _factor(x, factors):
            if x == 1:
                return
            if is_prime(x):
                factors.append(x)
            else:
                d = pollard_rho(x)
                _factor(d, factors)
                _factor(x // d, factors)
        
        factors = []
        _factor(temp_n, factors)
        
        for f in factors:
            if f > p:
                return False
        return True
