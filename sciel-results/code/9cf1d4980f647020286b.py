import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # First 30 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    best_n = 2
    best_score = 0
    
    def get_divisors(prime_list):
        divs = [1]
        for p in prime_list:
            new_divs = [d * p for d in divs]
            divs.extend(new_divs)
        return divs
        
    def calc_score(prime_list, n):
        divs = get_divisors(prime_list)
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # DFS to find best subset of size k
    def dfs(idx, current_prod, current_selected, k):
        nonlocal best_n, best_score
        
        if time.time() > deadline:
            return
            
        if len(current_selected) == k:
            if current_prod <= 10**18:
                s = calc_score(current_selected, current_prod)
                if s > best_score:
                    best_score = s
                    best_n = current_prod
            return
            
        # Pruning: if we have picked enough primes to reach k, but not yet, 
        # we need to ensure we can pick k - len(current_selected) more primes
        # from the remaining primes.
        # Also, if current_prod is already too big, prune.
        if current_prod > 10**18:
            return
            
        # If remaining primes are not enough to fill k, prune
        if len(primes) - idx < k - len(current_selected):
            return
            
        # Try including primes[idx]
        # Optimization: If we include primes[idx], the product grows.
        # We can also skip primes[idx].
        
        # Branch 1: Include primes[idx]
        new_prod = current_prod * primes[idx]
        if new_prod <= 10**18:
            current_selected.append(primes[idx])
            dfs(idx + 1, new_prod, current_selected, k)
            current_selected.pop()
            
        # Branch 2: Skip primes[idx]
        # Only if we can still fill the required number of primes
        if len(primes) - (idx + 1) >= k - len(current_selected):
            dfs(idx + 1, current_prod, current_selected, k)

    # Search for size 15
    dfs(0, 1, [], 15)
    
    # Search for size 14
    if time.time() < deadline - 10:
        dfs(0, 1, [], 14)

    return best_n
