import math
import heapq
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Primes up to 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    LIMIT = 10**18
    
    def calc_score(n):
        if n < 2:
            return 0
        # Factorize n
        temp = n
        factors = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
        if temp > 1:
            factors.append((temp, 1))
        
        # Generate divisors
        divs = [1]
        for p, exp in factors:
            new_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs += new_divs
        
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Initial best: Primorial of first 15 primes
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47
    best_n = 1
    best_score = 0
    best_exps = [1]*15 + [0]*10 # Assume we consider first 25 primes, but only first 15 are 1
    
    # Let's define the search space over the first 10 primes
    # Exponents for 2, 3, 5, 7, 11, 13, 17, 19, 23, 29
    # Primes 31 and above are likely 0 or 1.
    
    # We will use a priority queue.
    # State: (negative_tau, n, exps_tuple)
    # We start with the primorial of first 15 primes.
    
    start_exps = [1]*15 + [0]*10
    start_n = 1
    for i, e in enumerate(start_exps):
        start_n *= primes[i] ** e
        
    if start_n > LIMIT:
        # Reduce if necessary, but 15 primes is safe
        pass
        
    best_n = start_n
    best_score = calc_score(start_n)
    
    heap = []
    heapq.heappush(heap, (-1, start_n, tuple(start_exps)))
    visited = set()
    visited.add(tuple(start_exps))
    
    # We only care about the first 15 primes for the search.
    # If we include primes > 47, the product grows too fast.
    # Let's limit search to first 15 primes.
    
    while heap and time.time() < deadline:
        neg_tau, n, exps = heapq.heappop(heap)
        
        # Check neighbors
        for i in range(15):
            # Try increasing exponent
            new_exps = list(exps)
            new_exps[i] += 1
            new_n = n // (primes[i] ** exps[i]) * (primes[i] ** (new_exps[i]))
            # Optimization: calculate new_n directly
            new_n = n * primes[i]
            if new_n <= LIMIT:
                key = tuple(new_exps)
                if key not in visited:
                    visited.add(key)
                    # Estimate tau
                    tau = 1
                    for j in range(15):
                        tau *= (new_exps[j] + 1)
                    heapq.heappush(heap, (-tau, new_n, key))
                    
            # Try decreasing exponent
            if exps[i] > 0:
                new_exps = list(exps)
                new_exps[i] -= 1
                new_n = n // primes[i]
                if new_n >= 2:
                    key = tuple(new_exps)
                    if key not in visited:
                        visited.add(key)
                        tau = 1
                        for j in range(15):
                            tau *= (new_exps[j] + 1)
                        heapq.heappush(heap, (-tau, new_n, key))

        # Evaluate current node if it's promising
        # We evaluate the top candidates. 
        # To save time, we only evaluate if tau is high.
        # Or we can just evaluate everything popped, as the heap size is limited by time.
        
        # Let's evaluate the popped node.
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n

    return best_n
