import time
from fractions import Fraction
import sys
import numpy as np

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin for 150s limit
    
    primes = _primes_up_to(1000)
    # Sort primes ascending. Smaller primes are denser.
    primes.sort()
    
    MAX_L = 100000
    # covered[i] is True if i+1 is covered.
    # We care about 1..MAX_L.
    covered = np.zeros(MAX_L, dtype=bool)
    
    budget_remaining = Fraction(1, 1)
    used_primes = set()
    classes = []
    
    current_L = 0
    
    # Pre-calculate costs
    prime_costs = {p: Fraction(1, p) for p in primes}
    
    while time.time() < DEADLINE:
        if current_L >= MAX_L:
            break
            
        n = current_L + 1
        
        # Find the best prime to cover n
        # Heuristic: Smallest prime p (densest) that is unused and fits in budget
        # This tends to fill in gaps quickly.
        best_p = None
        for p in primes:
            if p in used_primes:
                continue
            cost = prime_costs[p]
            if budget_remaining - cost < 0:
                # Since primes are sorted ascending, if a small prime doesn't fit, 
                # larger ones (smaller cost) might fit. 
                # So we can't break here if we are looking for smallest p.
                # But if we are looking for smallest p, we just skip if it doesn't fit.
                continue
            
            # Check if p can cover n
            # Any p can cover n by choosing r = n % p.
            # So any unused, affordable prime is a candidate.
            # We want the one that helps extend L the most.
            # Smallest p is a good proxy for density.
            best_p = p
            break # Take the first (smallest) one that fits
            
        if best_p is None:
            # No prime fits or all used
            break
            
        # Commit to best_p
        r = n % best_p
        classes.append([best_p, r])
        used_primes.add(best_p)
        budget_remaining -= prime_costs[best_p]
        
        # Update covered array
        # We need to mark indices i such that (i+1) % p == r
        # i.e., i % p == (r - 1) % p
        start_idx = (r - 1) % p
        # Mark covered[start_idx :: p] = True
        covered[start_idx::best_p] = True
        
        # Advance current_L
        # We know n is covered. Check forward.
        # Using numpy to find the first False after current_L
        # But current_L might jump a lot.
        # Let's just loop while covered is True.
        # To speed up, we can check chunks or use numpy searchsorted on a derived array, 
        # but simple loop is O(gap size). Gaps should be small if strategy is good.
        while current_L < MAX_L and covered[current_L]:
            current_L += 1
            
    return classes

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Classes: {len(res)}")
    # Calculate L
    L_max = 100000
    is_covered = bytearray(L_max + 1)
    for p, r in res:
        start = r
        if start == 0:
            start = p
        for i in range(start, L_max + 1, p):
            is_covered[i] = 1
            
    L = 0
    while L < L_max and is_covered[L]:
        L += 1
    print(f"L: {L-1}")
