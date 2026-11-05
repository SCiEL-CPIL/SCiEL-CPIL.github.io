def construct():
    from math import factorial
    import time
    
    target = factorial(12)
    start_time = time.time()
    deadline = start_time + 85
    
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11 = 479001600
    primes = [2, 3, 5, 7, 11]
    total_exp = (10, 5, 2, 1, 1)
    
    best = [target]
    best_max = target
    
    def divisors_from_exp(exp):
        divs = [1]
        for i, e in enumerate(exp):
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= primes[i]
            divs = new_divs
        return divs
    
    def dfs(remaining_exp, current_factors, min_factor, depth):
        nonlocal best, best_max
        
        if time.time() > deadline:
            return
        
        if all(e == 0 for e in remaining_exp):
            if current_factors:
                mx = max(current_factors)
                if mx < best_max:
                    best_max = mx
                    best = list(current_factors)
            return
        
        # If only one more factor allowed
        if depth == 11:
            rem_val = 1
            for i, e in enumerate(remaining_exp):
                rem_val *= primes[i] ** e
            if rem_val > 12 and rem_val >= min_factor and rem_val not in current_factors:
                current_factors.append(rem_val)
                mx = max(current_factors)
                if mx < best_max:
                    best_max = mx
                    best = list(current_factors)
                current_factors.pop()
            return
        
        # Generate divisors of remaining value
        rem_divs = divisors_from_exp(remaining_exp)
        
        used = set(current_factors)
        candidates = [d for d in rem_divs if d >= min_factor and d > 12 and d not in used]
        candidates.sort()
        
        if not candidates:
            return
        
        rem_val = 1
        for i, e in enumerate(remaining_exp):
            rem_val *= primes[i] ** e
        
        for f in candidates:
            if f > rem_val:
                break
            
            new_exp = list(remaining_exp)
            n = f
            for i, p in enumerate(primes):
                while n % p == 0:
                    new_exp[i] -= 1
                    n //= p
            
            next_min = max(f + 1, 13)
            new_rem_val = rem_val // f
            
            if new_rem_val > 1:
                if new_rem_val < next_min:
                    continue
            
            current_factors.append(f)
            dfs(tuple(new_exp), current_factors, next_min, depth + 1)
            current_factors.pop()
            
            if time.time() > deadline:
                return
    
    dfs(total_exp, [], 13, 0)
    
    return best
