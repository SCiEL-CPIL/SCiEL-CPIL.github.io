import time
import random
import numpy as np

N = 5000
_DEADLINE = time.time() + 310.0  # safety margin before 330s kill

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

PRIMES = _primes(N)

def _greedy_sieve(active, primes, rng):
    """
    Greedy sieve: for each prime p <= current_size, remove the residue class
    with minimum active count. Returns the final active array.
    """
    active = active.copy()
    current_size = int(active.sum())
    
    for p in primes:
        if p > current_size:
            break
        
        # Count active in each residue class mod p
        residues = np.where(active)[0] % p
        counts = np.bincount(residues, minlength=p)
        
        # Find min count
        min_count = counts.min()
        
        # Find all classes with min count
        candidates = np.where(counts == min_count)[0]
        
        # Randomly pick one
        r = rng.choice(candidates)
        
        # Deactivate all active elements in class r mod p
        mask = (np.arange(N + 1) % p == r) & active
        active[mask] = False
        current_size -= int(mask.sum())
        
        if current_size == 0:
            break
    
    return active

def _local_improve(active, primes, rng, max_iter=50):
    """
    Try to improve by re-choosing residues for individual primes.
    For each prime, try all residue choices and keep the one that maximizes survivors.
    """
    active = active.copy()
    current_size = int(active.sum())
    
    # Rebuild the constraint set: we need to re-derive which primes are active
    # The primes that matter are those p <= current_size
    relevant_primes = [p for p in primes if p <= current_size]
    
    for _ in range(max_iter):
        improved = False
        for p in relevant_primes:
            # Try each residue class for prime p
            best_size = current_size
            best_r = None
            
            # Current residue for p (we need to track this)
            # Actually, let's just try all residues and see which gives max survivors
            # given the other constraints
            
            # For each candidate residue r for prime p:
            # Remove all x where x%p == r from active
            # Count remaining
            for r in range(p):
                test_active = active.copy()
                mask = (np.arange(N + 1) % p == r)
                test_active[mask] = False
                new_size = int(test_active.sum())
                if new_size > best_size:
                    best_size = new_size
                    best_r = r
            
            if best_r is not None and best_size > current_size:
                mask = (np.arange(N + 1) % p == best_r)
                active[mask] = False
                current_size = best_size
                improved = True
                # Update relevant primes
                relevant_primes = [q for q in primes if q <= current_size]
        
        if not improved:
            break
    
    return active

def construct():
    start_time = time.time()
    deadline = start_time + 310.0
    
    best_size = 0
    best_points = []
    
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    # Precompute arange for speed
    arange = np.arange(N + 1)
    
    # Strategy 1: Multiple randomized greedy sieves
    attempts = 0
    while time.time() < deadline - 30:  # leave 30s for local improvement
        active = np.ones(N + 1, dtype=bool)
        
        # Randomized greedy sieve
        active = _greedy_sieve(active, PRIMES, rng)
        
        points = arange[active].tolist()
        sz = len(points)
        
        if sz > best_size:
            best_size = sz
            best_points = points
        
        attempts += 1
        # Change RNG seed for diversity
        rng = random.Random(attempts * 7 + 13)
        np_rng = np.random.default_rng(attempts * 13 + 7)
    
    # Strategy 2: Local improvement on the best result
    if time.time() < deadline - 5 and best_size > 0:
        active = np.zeros(N + 1, dtype=bool)
        for x in best_points:
            active[x] = True
        
        improved_active = _local_improve(active, PRIMES, rng, max_iter=20)
        improved_points = arange[improved_active].tolist()
        
        if len(improved_points) > best_size:
            best_size = len(improved_points)
            best_points = improved_points
    
    # Strategy 3: Try a few more greedy sieves with different prime orderings
    while time.time() < deadline - 2:
        active = np.ones(N + 1, dtype=bool)
        
        # Shuffle primes for different removal order
        shuffled_primes = PRIMES.copy()
        rng.shuffle(shuffled_primes)
        
        # But we still need to process in a way that p <= current_size
        # Let's just do the greedy with shuffled tie-breaking
        current_size = N + 1
        for p in PRIMES:  # process in order, but random tie-breaking
            if p > current_size:
                break
            residues = arange[active] % p
            counts = np.bincount(residues, minlength=p)
            min_count = counts.min()
            candidates = np.where(counts == min_count)[0]
            r = rng.choice(candidates)
            mask = (arange % p == r) & active
            active[mask] = False
            current_size -= int(mask.sum())
            if current_size == 0:
                break
        
        points = arange[active].tolist()
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
    
    return best_points
