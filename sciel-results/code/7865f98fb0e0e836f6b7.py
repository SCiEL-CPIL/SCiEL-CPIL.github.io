def is_p_smooth(n, p, small_primes):
    if n == 1: return True
    # Trial divide by small primes
    temp = n
    for q in small_primes:
        if q * q > temp: break
        if q > p: break # If q > p, we stop? No, we need to remove all q <= p.
                        # But if q > p, we stop trial division.
                        # Wait, if q > p, we don't divide by q.
                        # So we only divide by q <= p.
        if temp % q == 0:
            while temp % q == 0:
                temp //= q
    if temp == 1:
        return True
    # Now temp has no prime factors <= min(p, 10^6)
    # If p >= 10^6, temp has no prime factors <= p.
    # So if temp > 1, it has a prime factor > p. Fail.
    if p >= 10**6:
        return False
    # If p < 10^6, we stopped at q > p or q^2 > temp.
    # If we stopped at q > p, then temp has no factors <= p.
    # So if temp > 1, it has a factor > p. Fail.
    # Wait, the loop `for q in small_primes` goes up to 10^6.
    # If p < 10^6, we should stop dividing at q > p.
    # So:
    for q in small_primes:
        if q > p: break
        if q * q > temp: break
        if temp % q == 0:
            while temp % q == 0:
                temp //= q
    if temp == 1:
        return True
    # If temp > 1, it must be prime (since we checked up to sqrt(temp) or p).
    # If we broke because q > p, then temp has no factors <= p.
    # So temp is a prime > p (or composite with factors > p).
    # In either case, fail.
    # If we broke because q^2 > temp, then temp is prime.
    # We need to check if temp <= p.
    if temp <= p:
        return True
    else:
        return False
