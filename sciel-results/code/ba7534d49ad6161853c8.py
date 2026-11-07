sigma = np.zeros(N + 1, dtype=np.int64)
sigma[1] = 1
primes = []
is_composite = np.zeros(N + 1, dtype=np.bool_)
p_pow = np.ones(N + 1, dtype=np.int64) # p^k for the smallest prime factor
# Linear sieve logic is complex to vectorize.
