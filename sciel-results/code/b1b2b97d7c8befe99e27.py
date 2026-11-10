import time
from math import gcd

def construct():
        start_time = time.time()
        deadline = start_time + 85.0 # 5s margin
        
        # 1. Primes and V_p
        n = 256
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        primes = [i for i in range(2, n+1) if is_prime[i]]
        
        V = {}
        for p in primes:
            max_val = 1
            while max_val * p <= n:
                max_val *= p
            V[p] = max_val
            
        # 2. Calculate Gain for each x
        candidates = [] # (gain, x, set_of_primes)
        for x in range(2, n + 1):
            # find prime factors
            temp = x
            factors = set()
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    factors.add(p)
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                factors.add(temp)
                
            cost = sum(V[p] for p in factors)
            gain = x - cost
            if gain > 0:
                candidates.append((gain, x, factors))
                
        # 3. Solve MWIS
        # Sort by gain descending
        candidates.sort(key=lambda t: t[0], reverse=True)
        
        # Simple backtracking
        best_gain = 0
        best_set = set()
        
        def backtrack(idx, current_gain, used_primes, current_set):
            nonlocal best_gain, best_set
            if time.time() > deadline:
                return
            if current_gain > best_gain:
                best_gain = current_gain
                best_set = set(current_set)
                
            if idx == len(candidates):
                return
                
            # Pruning: if current_gain + sum of remaining gains <= best_gain, return
            # (Optional, but good for safety)
            
            # Option 1: Skip candidate idx
            backtrack(idx + 1, current_gain, used_primes, current_set)
            
            # Option 2: Take candidate idx
            gain, x, factors = candidates[idx]
            if not any(p in used_primes for p in factors):
                backtrack(idx + 1, current_gain + gain, used_primes | factors, current_set | {x})

        backtrack(0, 0, set(), set())
        
        # 4. Construct final list
        result = [1]
        used_primes_final = set()
        for x in best_set:
            result.append(x)
            # Mark primes used by x
            temp = x
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    used_primes_final.add(p)
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                used_primes_final.add(temp)
                
        for p in primes:
            if p not in used_primes_final:
                result.append(V[p])
                
        return result
