import time
import numpy as np
import random

N = 5000
_DEADLINE = time.time() + 320.0

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

def _greedy_solve(points, primes, heuristic="min", seed=0):
    """
    Greedy removal of residue classes.
    points: numpy array of current points
    primes: list of primes to consider
    heuristic: 'min', 'max', 'random_min'
    seed: random seed
    """
    points = points.copy()
    rng = random.Random(seed)
    
    # We need to iterate through primes. 
    # The condition is p <= len(points). 
    # As points shrinks, len(points) decreases, so the set of relevant primes shrinks.
    # We can just iterate through all primes <= initial_len and check if p <= current_len.
    
    for p in primes:
        if time.time() > _DEADLINE - 5:
            break
            
        k = len(points)
        if p > k:
            continue
            
        # Count residues
        # points % p
        residues = points % p
        # np.bincount is fast
        counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(counts)
        
        if min_count > 0:
            # Need to remove a class
            if heuristic == "min":
                # Find all classes with min count
                candidates = np.where(counts == min_count)[0]
                # Pick one. To introduce variation, we can pick randomly from candidates
                target_residue = candidates[rng.randint(0, len(candidates) - 1)]
            elif heuristic == "max":
                candidates = np.where(counts == np.max(counts))[0]
                target_residue = candidates[rng.randint(0, len(candidates) - 1)]
            elif heuristic == "random":
                # Pick random non-zero class
                candidates = np.where(counts > 0)[0]
                target_residue = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                target_residue = 0
                
            # Remove points where v % p == target_residue
            mask = (points % p) != target_residue
            points = points[mask]
            
    return points

def construct():
    points = np.arange(N + 1)
    # Primes up to N
    primes = _primes(N)
    
    best_points = None
    best_size = -1
    
    # Try different heuristics and orders
    # 1. Standard increasing primes, min count
    # 2. Standard increasing primes, random min count
    # 3. Decreasing primes? (Unlikely to be better, but worth a try if time permits)
    
    heuristics = ["min", "random_min", "min"] # Duplicates to allow random variation
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # We will run a few iterations
    for h in heuristics:
        for s in seeds:
            if time.time() > _DEADLINE - 10:
                break
                
            # Create a copy of the heuristic string to modify if needed, 
            # but "random_min" isn't in the function yet. Let's add it.
            # Actually, I defined 'min' with random tie-breaking in the function above.
            # So 'min' is already randomized if I use different seeds.
            
            res = _greedy_solve(points, primes, heuristic=h, seed=s)
            
            # Check validity
            k = len(res)
            valid = True
            # Check primes up to k
            for p in primes:
                if p > k:
                    break
                residues = res % p
                counts = np.bincount(residues, minlength=p)
                if np.min(counts) > 0:
                    valid = False
                    break
            
            if valid and k > best_size:
                best_size = k
                best_points = res.copy()
                
    if best_points is None:
        # Fallback to original logic if something went wrong
        return list(range(N + 1))
        
    return list(best_points)
