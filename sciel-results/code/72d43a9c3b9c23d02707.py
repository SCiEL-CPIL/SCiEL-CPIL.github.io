def phi_fast(n):
    if n <= 1:
        return 0
    result = n
    y = n
    for p in primes_list:
        if p * p > y:
            break
        if y % p == 0:
            result -= result // p
            while y % p == 0:
                y //= p
    if y > 1:
        result -= result // y
    return result
