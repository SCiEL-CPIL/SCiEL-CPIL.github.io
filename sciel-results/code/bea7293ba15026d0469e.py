def construct():
          N = 256
          # Sieve primes
          is_prime = [True] * (N + 1)
          is_prime[0] = is_prime[1] = False
          for i in range(2, int(N**0.5) + 1):
              if is_prime[i]:
                  for j in range(i*i, N + 1, i):
                      is_prime[j] = False
          
          primes = [i for i in range(2, N + 1) if is_prime[i]]
          
          result = [1]
          for p in primes:
              power = p
              while power * p <= N:
                  power *= p
              result.append(power)
          
          return result
