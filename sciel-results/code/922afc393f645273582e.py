import math
import time

def construct():
            N = 1000000
            LO = 10000
            deadline = time.time() + 85.0
            
            # 1. Sieve of Eratosthenes to get primes
            is_prime = [True] * (N + 2)
            is_prime[0] = is_prime[1] = False
            primes = []
            for i in range(2, N + 2):
                if is_prime[i]:
                    primes.append(i)
                    for j in range(i * i, N + 2, i):
                        is_prime[j] = False
            
            # 2. Initialize powerful part
            pp = [1] * (N + 2)
            
            # 3. For each prime, update multiples
            for p in primes:
                p2 = p * p
                if p2 > N:
                    break
                # Find max power p^k <= N
                p_power = p2
                k = 2
                while p_power * p <= N:
                    p_power *= p
                    k += 1
                
                # We need to assign p^k to m if v_p(m) = k
                # Iterate through powers
                current_power = p2
                exp = 2
                while current_power <= N:
                    next_power = current_power * p
                    # Multiples of current_power
                    for m in range(current_power, N + 2, current_power):
                        # Check if m is divisible by next_power
                        if m % next_power != 0:
                            pp[m] *= current_power
                    current_power = next_power
                    exp += 1
            
            # 4. Find max product
            best_n = LO
            best_val = pp[LO] * pp[LO + 1]
            
            for n in range(LO, N + 1):
                val = pp[n] * pp[n + 1]
                if val > best_val:
                    best_val = val
                    best_n = n
                if (n - LO) % 100000 == 0 and time.time() > deadline:
                    break
            
            return best_n
