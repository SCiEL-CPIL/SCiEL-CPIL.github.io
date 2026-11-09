import time
import random
from collections import defaultdict

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_prime(n, primes_set):
    return n in primes_set

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s buffer
    
    all_primes = _primes(N)
    primes_set = set(all_primes)
    
    best_size = 0
    best_points = []
    
    # Restart loop
    while time.time() < deadline:
        # Phase 1: Greedy Removal
        points = list(range(N))
        points_set = set(points)
        
        # Initialize counts
        # counts[p] is a list of length p
        counts = {p: [0] * p for p in all_primes}
        for v in points:
            for p in all_primes:
                counts[p][v % p] += 1
                
        current_size = len(points)
        
        # Remove to make admissible
        while time.time() < deadline:
            bad_prime = None
            min_count = float('inf')
            target_class = -1
            
            # Find a prime p <= current_size that is fully covered
            # and has the minimum count in its smallest class
            for p in all_primes:
                if p > current_size:
                    break
                m = min(counts[p])
                if m > 0:
                    if m < min_count:
                        min_count = m
                        bad_prime = p
                        for r, c in enumerate(counts[p]):
                            if c == m:
                                target_class = r
                                break
            
            if bad_prime is None:
                break
            
            # Find an element to remove
            x_to_remove = None
            for x in points:
                if x % bad_prime == target_class:
                    x_to_remove = x
                    break
            
            if x_to_remove is None:
                break
            
            # Remove
            points.remove(x_to_remove)
            points_set.discard(x_to_remove)
            current_size -= 1
            
            # Update counts
            for p in all_primes:
                counts[p][x_to_remove % p] -= 1
                
        # Phase 2: Add Back
        removed = [v for v in range(N) if v not in points_set]
        random.shuffle(removed)
        
        # Compute missing residues for current points
        # missing[p] = set of residues not in points mod p
        missing = {}
        n = len(points)
        for p in all_primes:
            if p > n:
                break
            residues = set()
            for v in points:
                residues.add(v % p)
            missing[p] = set(range(p)) - residues
            
        # Try to add back
        for v in removed:
            if time.time() > deadline:
                break
            
            new_n = n + 1
            
            # Check if adding v violates admissibility
            ok = True
            
            # 1. For all p <= n, v % p must not be the unique missing residue
            for p in all_primes:
                if p > n:
                    break
                if v % p in missing[p]:
                    if len(missing[p]) == 1:
                        ok = False
                        break
            
            if not ok:
                continue
            
            # 2. Check p = new_n if prime
            if new_n in primes_set:
                # Check if S \cup {v} covers all residues mod new_n
                # S has n elements. new_n = n+1.
                # If S has n distinct residues mod new_n, and v has the missing one, it's bad.
                residues_new = set()
                for v2 in points:
                    residues_new.add(v2 % new_n)
                if len(residues_new) == n:
                    missing_res = (set(range(new_n)) - residues_new).pop()
                    if v % new_n == missing_res:
                        ok = False
            
            if not ok:
                continue
            
            # Add v
            points.append(v)
            points_set.add(v)
            n = new_n
            
            # Update missing sets
            for p in all_primes:
                if p > n:
                    break
                r = v % p
                if r in missing[p]:
                    missing[p].discard(r)
                    
        # Update best
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            
        # If we are running out of time, break
        if time.time() > deadline - 20:
            break

    return best_points
