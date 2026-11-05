import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = math.factorial(12)
    MIN_F = 13
    MAX_FACTORS = 12
    
    # Compute all divisors of N efficiently
    def compute_divisors(n):
        divs = {1}
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                new_divs = set()
                power = 1
                for e in range(exp + 1):
                    for d in divs:
                        new_divs.add(d * power)
                    power *= p
                divs = new_divs
            p += 1 if p == 2 else 2
        if temp > 1:
            new_divs = set()
            for d in divs:
                new_divs.add(d)
                new_divs.add(d * temp)
            divs = new_divs
        return sorted(divs)
    
    all_divs = compute_divisors(N)
    valid_divs = [d for d in all_divs if d > MIN_F]
    valid_divs_set = set(valid_divs)
    n_valid = len(valid_divs)
    
    # Precompute lower-bound product table
    # lb_prod[i][k] = product of k smallest valid divisors starting at index i
    max_k = MAX_FACTORS
    INF = float('inf')
    lb_prod = [[INF] * (max_k + 1) for _ in range(n_valid + 1)]
    for i in range(n_valid, -1, -1):
        lb_prod[i][0] = 1
        for k in range(1, max_k + 1):
            if i + k > n_valid:
                lb_prod[i][k] = INF
            else:
                prev = lb_prod[i][k-1]
                if prev == INF:
                    lb_prod[i][k] = INF
                else:
                    val = prev * valid_divs[i + k - 1]
                    lb_prod[i][k] = val if val <= N else INF
    
    # Initial best: single factor N
    best_max = N
    best_factors = [N]
    
    # Prime check for fast handling
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def can_complete(remaining, min_idx, k):
        """Check if remaining can be factored into k distinct divisors >= valid_divs[min_idx]."""
        if k == 0:
            return remaining == 1
        if remaining < 1:
            return False
        if min_idx >= n_valid:
            return False
        if lb_prod[min_idx][k] > remaining:
            return False
        if remaining < valid_divs[min_idx]:
            return False
        if k == 1:
            return remaining in valid_divs_set and remaining >= valid_divs[min_idx]
        return True
    
    def dfs(remaining, min_idx, k, current_max, current_factors):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
        
        if k == 0:
            if remaining == 1:
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(current_factors)
            return
        
        if remaining < 1 or min_idx >= n_valid:
            return
        
        # Pruning: lower bound
        if lb_prod[min_idx][k] > remaining:
            return
        
        if remaining < valid_divs[min_idx]:
            return
        
        # If k == 1, remaining must be a valid divisor
        if k == 1:
            if remaining in valid_divs_set and remaining >= valid_divs[min_idx]:
                new_max = max(current_max, remaining)
                if new_max < best_max:
                    best_max = new_max
                    best_factors = current_factors + [remaining]
            return
        
        # Try each possible next factor starting from min_idx
        # To find good solutions fast, try factors that keep the remainder balanced
        # Sort candidates by how close they are to sqrt(remaining)
        candidates = []
        for i in range(min_idx, n_valid):
            f = valid_divs[i]
            if f > remaining:
                break
            if remaining % f != 0:
                continue
            new_max = max(current_max, f)
            if new_max >= best_max:
                break
            new_remaining = remaining // f
            if not can_complete(new_remaining, i + 1, k - 1):
                continue
            candidates.append((abs(f - math.sqrt(remaining)), f, i))
        
        # Sort by balance (closest to sqrt first), then by value
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        for _, f, i in candidates:
            new_max = max(current_max, f)
            if new_max >= best_max:
                break
            
            new_remaining = remaining // f
            current_factors.append(f)
            dfs(new_remaining, i + 1, k - 1, new_max, current_factors)
            current_factors.pop()
            
            # Early termination if we found a very good solution
            if best_max <= MIN_F + 10:
                return
    
    # Try different numbers of factors
    # Start with smaller k (fewer factors, each larger) then increase
    for k in range(2, MAX_FACTORS + 1):
        if time.time() > deadline:
            break
        
        # Check if k factors is even possible
        if lb_prod[0][k] > N:
            break
        
        dfs(N, 0, k, MIN_F - 1, [])
        
        # If we found a solution with max very close to theoretical minimum, stop
        if best_max <= MIN_F + k + 5:
            break
    
    return best_factors
