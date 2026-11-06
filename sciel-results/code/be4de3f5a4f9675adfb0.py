import math

def construct():
        N = 1000000
        # Sieve for primes
        is_prime = [True] * (N + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(N)) + 1):
            if is_prime[i]:
                for j in range(i*i, N + 1, i):
                    is_prime[j] = False
        primes = [i for i, v in enumerate(is_prime) if v]
        
        P = [1] * (N + 1)
        
        for p in primes:
            power = p * p
            if power > N:
                continue
            next_power = power * p
            while power <= N:
                # If next_power > N, all multiples of power in [1, N] are NOT divisible by next_power
                if next_power > N:
                    for m in range(power, N + 1, power):
                        P[m] *= power
                else:
                    for m in range(power, N + 1, power):
                        if m % next_power != 0:
                            P[m] *= power
                power = next_power
                next_power = power * p
                if power > N:
                    break
                    
        best_n = 10000
        best_val = -1
        
        for n in range(10000, N):
            val = P[n] * P[n+1]
            if val > best_val:
                best_val = val
                best_n = n
                
        return best_n
