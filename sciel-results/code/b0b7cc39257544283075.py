def construct():
    # The smallest n with phi(n) divisible by 120 is 143 (phi(143)=120).
    # We verify this with a small sieve to stay within the same algorithmic family.
    N = 200
    phi = list(range(N + 1))
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    return 1000000
