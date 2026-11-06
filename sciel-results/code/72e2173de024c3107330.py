def is_p_smooth(n, p):
    if n == 1:
        return True
    # If n <= p, all prime factors are <= n <= p
    if n <= p:
        return True
    # Trial division by primes up to min(p, some_limit)
    for q in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if q > p:
            break
        while n % q == 0:
            n //= q
    if n == 1:
        return True
    if is_prime(n):
        return n <= p
    # Factor remaining n using Pollard rho
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if m > p:
                return False
        else:
            d = pollard_rho(m)
            stack.append(d)
            stack.append(m // d)
    return True
