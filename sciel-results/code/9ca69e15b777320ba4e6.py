    large_part = list(range(max_val + 1))
    for p in small_primes:
        for multiple in range(p, max_val + 1, p):
            while large_part[multiple] % p == 0:
                large_part[multiple] //= p
    