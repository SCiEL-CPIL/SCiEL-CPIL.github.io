import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    def sieve(n):
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    num_primes = len(primes)
    
    # Target: cover as much of 1..L as possible
    # We'll use a greedy approach with budget tracking
    
    # For each prime p, the possible residues are 0..p-1
    # A class (p, r) covers all n where n % p == r
    
    # Greedy set cover with budget:
    # State: which integers 1..MAX_L are covered
    # MAX_L: we'll try to go up to 100000 but focus on contiguous from 1
    
    MAX_L = 100000
    covered = np.zeros(MAX_L + 1, dtype=bool)  # covered[n] for n in 1..MAX_L
    
    # Track budget using Fraction for exactness
    budget_used = Fraction(0, 1)
    budget_limit = Fraction(1, 1)
    
    # For each prime, precompute which residues are "available" (not yet used)
    # Actually, we can use each prime at most once (distinct primes)
    used_primes = set()
    
    # Greedy: repeatedly find the class that covers the most new integers
    # among those that cover the current frontier (smallest uncovered)
    
    best_L = 0
    classes = []
    
    # Precompute for each prime p, the coverage pattern
    # For efficiency, we'll work with the frontier approach
    
    # Frontier: the smallest uncovered integer
    def find_frontier(covered, MAX_L):
        # Find smallest n >= 1 not covered
        for n in range(1, MAX_L + 1):
            if not covered[n]:
                return n
        return MAX_L + 1
    
    # For a given frontier f, find best (p, r) where r = f % p
    # and p not used, and budget allows
    # Score: number of new integers covered in [1, MAX_L]
    
    # To speed up, precompute coverage counts
    # For prime p and residue r, the numbers covered are r, r+p, r+2p, ...
    # In range [1, MAX_L]: count = floor((MAX_L - r) / p) if r >= 1, etc.
    
    # Actually, let's just do it directly but efficiently
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        frontier = find_frontier(covered, MAX_L)
        if frontier > MAX_L:
            break
        
        # Find best (p, r) to cover frontier
        # r must equal frontier % p
        best_gain = -1
        best_p = None
        best_r = None
        
        for p in primes:
            if p in used_primes:
                continue
            cost = Fraction(1, p)
            if budget_used + cost > budget_limit:
                continue
            
            r = frontier % p
            
            # Count how many new integers this would cover
            # Numbers: r, r+p, r+2p, ... up to MAX_L
            # But r could be 0, in which case it's p, 2p, 3p, ...
            if r == 0:
                first = p
            else:
                first = r
            
            if first > MAX_L:
                continue
            
            # Count new covers
            gain = 0
            # Use numpy for speed
            nums = np.arange(first, MAX_L + 1, p)
            new_covers = np.sum(~covered[nums])
            
            if new_covers > best_gain:
                best_gain = new_covers
                best_p = p
                best_r = r
        
        if best_p is None:
            break
        
        # Apply the class
        p = best_p
        r = best_r
        used_primes.add(p)
        budget_used += Fraction(1, p)
        classes.append([p, r])
        
        # Update covered
        if r == 0:
            first = p
        else:
            first = r
        
        if first <= MAX_L:
            covered[first::p] = True
        
        # Update best_L
        # Find how far contiguous coverage goes
        # We can track this incrementally
        while best_L < MAX_L and covered[best_L + 1]:
            best_L += 1
        
        if best_L >= 999:
            # We've achieved the target, but let's see if we can do better
            # Continue for a bit more
            pass
    
    return classes
