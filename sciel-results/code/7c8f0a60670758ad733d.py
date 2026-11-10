# Compute M(x) for all x in [1, max_x]
# M(x) = sum_{n=1}^{x} g(n) where g(n) = sum_{d|n, d<=K} mu(d)
# For x <= K: M(x) = sum_{k=1}^{x} mu(k) floor(x/k) ... no
# M(x) = sum_{k=1}^{K} mu(k) floor(x/k)
# For x <= K: M(x) = sum_{k=1}^{x} mu(k) floor(x/k) since floor(x/k)=0 for k>x
# For x > K: M(x) = sum_{k=1}^{K} mu(k) floor(x/k)

# g(n) = sum_{d|n, d<=K} mu(d)
# M(x) = sum_{n=1}^{x} g(n)

# So compute g(n) for n=1..max_x, then prefix sum to get M(x)
