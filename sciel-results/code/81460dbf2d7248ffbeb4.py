N = 1000000
pp = [1] * (N + 2)
primes = get_primes(1000)
for p in primes:
    p2 = p * p
    # Powers of p
    # We need to iterate multiples of p^2, p^3, etc?
    # No, for each m multiple of p^2, we find the full power of p in m.
    # So we just iterate m = p^2, 2*p^2, 3*p^2 ...
    # This covers all m with v_p(m) >= 2.
    for m in range(p2, N + 1, p2):
        x = m
        e = 2
        while x % p == 0:
            x //= p
            e += 1
        pp[m] *= (p ** e)
