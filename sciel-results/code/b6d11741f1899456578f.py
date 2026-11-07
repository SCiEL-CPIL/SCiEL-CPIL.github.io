def compute_sigma(N):
    sigma = [1] * (N + 1)
    # For each i, add i to sigma[m] for all multiples m of i
    # This is O(N log N) but with good cache behavior
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    return sigma
