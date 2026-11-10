    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
        if i * i <= K:
            for j in range(i * i, K + 1, i * i):
                mu[j] = 0
