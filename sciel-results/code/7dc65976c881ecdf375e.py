import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    
    L_max = 2000
    covered = np.zeros(L_max + 1, dtype=bool)
    
    selected = []
    used_primes = set()
    budget = Fraction(1, 1)
    
    while True:
        if time.time() > deadline:
            break
            
        # Determine k (smallest uncovered integer)
        if not covered[0]:
            k = 1
        else:
            # Check if any uncovered in 2..L_max
            if not np.any(~covered[1:]):
                break
            first_uncovered_idx = np.argmax(~covered[1:]) + 1
            k = first_uncovered_idx + 1
            
        # Find best prime
        best_p = -1
        best_r = -1
        best_score = -1
        
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if cost > budget:
                continue
                
            r = k % p
            
            if r == 0:
                start_idx = p - 1
            else:
                start_idx = r - 1
                
            if start_idx >= L_max:
                continue
                
            # Calculate gain
            gain = np.sum(~covered[start_idx : L_max : p])
            
            # We must cover k, so gain >= 1
            # Score: gain * p
            score = gain * p
            
            if score > best_score:
                best_score = score
                best_p = p
                best_r = r
                
        if best_p == -1:
            break
            
        selected.append([best_p, best_r])
        used_primes.add(best_p)
        budget -= Fraction(1, best_p)
        
        if best_r == 0:
            start_idx = best_p - 1
        else:
            start_idx = best_r - 1
            
        covered[start_idx : L_max : best_p] = True
        
    # Determine L
    if not np.any(~covered):
        L = L_max
    else:
        first_false_idx = np.argmax(~covered)
        L = first_false_idx
        
    return selected
