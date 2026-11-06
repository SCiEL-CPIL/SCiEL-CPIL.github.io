# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a linear-time sieve variant (Euler's sieve) to compute phi(n) for all n,
    then scans for the first n where phi(n) % 120 == 0.
    """
    N = 1000000
    # Initialize phi[n] = n for all n
    phi = [0] * (N + 1)
    phi[0] = 0
    phi[1] = 1
    for i in range(2, N + 1):
        phi[i] = i

    # Linear sieve for phi
    primes = []
    is_composite = [False] * (N + 1)

    for i in range(2, N + 1):
        if not is_composite[i]:
            primes.append(i)
            phi[i] = i - 1  # phi(p) = p - 1 for prime p
        for p in primes:
            if i * p > N:
                break
            is_composite[i * p] = True
            if i % p == 0:
                # p divides i, so phi(i*p) = phi(i) * p
                phi[i * p] = phi[i] * p
                break
            else:
                # p does not divide i, so phi(i*p) = phi(i) * (p - 1)
                phi[i * p] = phi[i] * (p - 1)

    # Find the smallest n with phi(n) divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n

    return 1
# EVOLVE-BLOCK-END
