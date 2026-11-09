def construct():
    n = 256
    # The condition for the interval [k, n] to be valid is 3*k > 2*n
    # We want to minimize k to maximize the size n - k + 1
    # k > 2*n / 3
    k = int(2 * n / 3) + 1
    return list(range(k, n + 1))
