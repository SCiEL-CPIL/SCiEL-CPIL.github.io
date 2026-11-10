import time
import random
import math

# EVOLVE-BLOCK-START

def _build_spf(limit):
    """Build Smallest Prime Factor array up to limit."""
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

# Max possible sum is 10000 + 9999 = 19999
_MAX_SUM = 20000
_SPF = _build_spf(_MAX_SUM)

def _prime_factors(n):
    """Return the set of prime factors of n."""
    factors = set()
    while n > 1:
        p = _SPF[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _score(arr):
    """Calculate the number of distinct prime factors of all pairwise sums."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(_prime_factors(s))
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Initial candidate: multiples of 3
    # Sums are 3*(i+j) for i,j in 1..12
    # i+j ranges from 3 to 23
    # Primes in 3..23: 3, 5, 7, 11, 13, 17, 19, 23
    # Total distinct primes: 8
    best_arr = [3 * i for i in range(1, 13)]
    best_score = _score(best_arr)
    
    # Local search to see if we can improve
    # We will try random restarts and local moves
    # Move: pick one element, change to a random distinct value <= 10000
    
    current_arr = best_arr[:]
    current_score = best_score
    
    # Random seed for reproducibility if needed, but random is fine
    random.seed(42)
    
    while time.time() < deadline:
        # Generate a neighbor
        new_arr = current_arr[:]
        idx = random.randint(0, 11)
        old_val = new_arr[idx]
        
        # Try to find a new value
        # Heuristic: keep values small to keep sums small
        # But we might need to jump to a different "cluster"
        
        # Strategy 1: Small perturbation
        # Strategy 2: Random restart to a different base
        
        if random.random() < 0.1:
            # Random restart: pick a new base prime or structure
            # Try multiples of a random small prime
            base_prime = random.choice([2, 3, 5, 7, 11, 13])
            # Generate 12 distinct multiples
            candidates = []
            for k in range(1, 200):
                val = base_prime * k
                if val <= 10000:
                    candidates.append(val)
            # Pick 12 random distinct ones? Or just the first 12?
            # First 12 are usually best for minimizing sum magnitude
            if len(candidates) >= 12:
                new_arr = candidates[:12]
                # Ensure distinct (they are by construction)
                # Check validity
                if len(set(new_arr)) == 12:
                    s = _score(new_arr)
                    if s < current_score:
                        current_score = s
                        current_arr = new_arr[:]
                    if s < best_score:
                        best_score = s
                        best_arr = new_arr[:]
        else:
            # Local move: change one element
            # Try a few random replacements
            for _ in range(10):
                trial_val = random.randint(1, 10000)
                if trial_val == old_val:
                    continue
                if trial_val in new_arr:
                    continue
                
                new_arr[idx] = trial_val
                s = _score(new_arr)
                if s < current_score:
                    current_score = s
                    current_arr = new_arr[:]
                    if s < best_score:
                        best_score = s
                        best_arr = current_arr[:]
                    break
                else:
                    new_arr[idx] = old_val
            
            # If no improvement in local move, maybe accept a worse move with prob (simulated annealing style)
            # But simple hill climbing is often enough for this small space if we restart often.
            # Let's add a simple random jump if stuck
            if random.random() < 0.05:
                 # Jump to random distinct set of small numbers
                 pool = list(range(1, 50))
                 random.shuffle(pool)
                 new_arr = pool[:12]
                 s = _score(new_arr)
                 if s < current_score:
                     current_score = s
                     current_arr = new_arr[:]
                     if s < best_score:
                         best_score = s
                         best_arr = current_arr[:]
    
    return best_arr

# EVOLVE-BLOCK-END
