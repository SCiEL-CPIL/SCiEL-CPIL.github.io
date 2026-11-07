import time
import random
from itertools import combinations

# Precompute prime exponent vectors for numbers 2 to 40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

def get_vector(n):
    vec = [0] * NUM_PRIMES
    temp = n
    for i, p in enumerate(PRIMES):
        while temp % p == 0:
            vec[i] += 1
            temp //= p
    return tuple(vec)

# Map number to its vector
VECTORS = {n: get_vector(n) for n in range(2, 41)}
CANDIDATES = list(range(2, 41))

def is_dissociated(vecs):
    """
    Check if the set of vectors is dissociated, i.e., all subset sums are distinct.
    """
    k = len(vecs)
    if k == 0:
        return True
    sums = set()
    # Use bit manipulation to generate subsets
    # For k up to 20, 2^k is manageable.
    # For k=15, 2^15 = 32768.
    for mask in range(1 << k):
        s = [0] * NUM_PRIMES
        # Optimization: build sum incrementally?
        # For small k, direct calculation is fine.
        for i in range(k):
            if mask & (1 << i):
                v = vecs[i]
                for j in range(NUM_PRIMES):
                    s[j] += v[j]
        if tuple(s) in sums:
            return False
        sums.add(tuple(s))
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    best_set = None
    best_size = 0
    
    # Initialize with primes (known valid set of size 12)
    primes_set = [p for p in CANDIDATES if VECTORS[p] == tuple([1 if p==prime else 0 for prime in PRIMES])]
    # Simpler: just check if prime
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
    primes_set = [n for n in CANDIDATES if is_prime(n)]
    if is_dissociated([VECTORS[n] for n in primes_set]):
        best_set = primes_set
        best_size = len(primes_set)
    else:
        best_set = []
        best_size = 0

    random.seed(42)
    
    while time.time() < deadline:
        # Random restart greedy
        random.shuffle(CANDIDATES)
        current_set = []
        current_vecs = []
        
        for n in CANDIDATES:
            if time.time() > deadline:
                break
            v = VECTORS[n]
            # Try adding n
            new_vecs = current_vecs + [v]
            if is_dissociated(new_vecs):
                current_set.append(n)
                current_vecs = new_vecs
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # If we found a very large set, we can try to polish it
            # But for now, just update best.
            
        # Optional: Local search on best_set if time permits
        # Try to swap one element for two others to increase size
        # This is complex, so we rely on many greedy restarts first.
        
    return best_set

# EVOLVE-BLOCK-END
