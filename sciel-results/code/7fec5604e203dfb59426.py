import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    max_n = 10**18
    
    candidates = []
    
    # DFS to generate numbers with non-increasing exponents
    # This generates Highly Composite Numbers and their neighbors
    def dfs(idx, current_n, current_tau, last_exp):
        if time.time() > deadline:
            return
        if idx >= len(primes):
            return
            
        p = primes[idx]
        # Max exponent for this prime given remaining budget
        # current_n * p^e <= max_n
        max_e = 0
        temp_n = current_n
        while temp_n * p <= max_n:
            temp_n *= p
            max_e += 1
            
        # Also constrained by last_exp to maintain non-increasing property
        # Actually, we can relax this slightly to find more candidates, 
        # but non-increasing is a strong heuristic for high tau.
        
        for e in range(min(max_e, last_exp), -1, -1):
            new_n = current_n * (p ** e)
            if new_n > max_n:
                continue
            new_tau = current_tau * (e + 1)
            
            # If we have assigned all primes we care about, add to candidates
            # We stop adding primes when p > max_n / current_n (e=0)
            # But we need to decide when to stop.
            # If e == 0, we can continue to next prime with last_exp=0? 
            # If e=0, subsequent exponents must be 0.
            
            if e == 0:
                # Add current_n as a candidate if it has at least 2 primes? 
                # Or just add it.
                if current_n > 1:
                    candidates.append((current_tau, current_n))
                # No need to recurse further if e=0 and we want non-increasing?
                # If e=0, next must be 0. So we are done with this branch.
                return
            
            dfs(idx + 1, new_n, new_tau, e)
            
            # If we break early due to time, just return
            if time.time() > deadline:
                return

    # Start DFS
    # Initial last_exp can be high, e.g., 60
    dfs(0, 1, 1, 60)
    
    # Sort candidates by tau descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for tau, n in candidates:
        if n not in seen:
            seen.add(n)
            unique_candidates.append((tau, n))
            
    # Evaluate top candidates for tau_perp
    best_score = 0
    best_n = 2
    
    # Check top 1000 candidates by tau
    for tau, n in unique_candidates[:1000]:
        if time.time() > deadline:
            break
            
        divisors = get_divisors(n)
        divisors.sort()
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
                
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n

def get_divisors(n):
    divs = [1]
    p = 2
    while p * p <= n:
        if n % p == 0:
            # Extract p^k
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                # Extend divisors
                new_divs = []
                for d in divs:
                    # multiply by p^j
                    temp = d
                    for _ in range(0, 0): # placeholder
                        pass
                    # This logic is flawed, better to generate from scratch or use a list
                # Let's use a simpler divisor generation
            # Restart with a cleaner divisor generator
        p += 1 if p == 2 else 2
        
    # Simpler divisor generator
    divs = [1]
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            pow = 1
            while temp % p == 0:
                temp //= p
                pow *= p
                new_divs = []
                for d in divs:
                    new_divs.append(d * pow)
                divs.extend(new_divs)
        p += 1 if p == 2 else 2
    if temp > 1:
        new_divs = []
        for d in divs:
            new_divs.append(d * temp)
        divs.extend(new_divs)
    return divs
