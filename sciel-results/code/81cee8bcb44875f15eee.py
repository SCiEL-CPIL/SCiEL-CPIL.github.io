import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    # Generate primes up to 1000
    primes = []
    for i in range(2, 1001):
        is_prime = True
        for j in range(2, int(i**0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    primes.sort()
    
    used_primes = set()
    total_cost = Fraction(0, 1)
    result = []
    
    # Max L is capped at 10^5
    MAX_L = 100000
    covered = [False] * (MAX_L + 1)
    
    # Index to find next unused prime quickly?
    # Since we always pick the smallest, we can just iterate.
    # But iterating from start every time is O(N^2). N~168. 168^2 is small.
    
    L = 0
    n = 1
    
    while n <= MAX_L:
        if time.time() > deadline:
            break
            
        if covered[n]:
            L = n
            n += 1
            continue
            
        # Find smallest unused prime
        chosen_prime = None
        for p in primes:
            if p in used_primes:
                continue
                
            # Check budget
            new_cost = total_cost + Fraction(1, p)
            if new_cost <= 1:
                chosen_prime = p
                break
            else:
                # Since primes are sorted, larger primes have smaller cost.
                # Wait, if smallest prime exceeds budget, larger ones might fit?
                # No, 1/p decreases as p increases.
                # So if 1/p_small > remaining budget, 1/p_large < 1/p_small might fit.
                # So we cannot break immediately. We must continue checking.
                # However, if the *largest* prime doesn't fit, we break.
                pass
                
        if chosen_prime is None:
            # No prime could be selected due to budget or no primes left
            # We need to find if ANY prime fits.
            # The loop above only breaks if it finds one.
            # If it finishes without finding one, no prime fits.
            break
            
        # Select chosen_prime
        total_cost += Fraction(1, chosen_prime)
        used_primes.add(chosen_prime)
        r = n % chosen_prime
        result.append([chosen_prime, r])
        
        # Update covered array
        # Mark n, n+p, n+2p... as covered
        for k in range(n, MAX_L + 1, chosen_prime):
            covered[k] = True
            
        L = n
        n += 1
        
    return result
