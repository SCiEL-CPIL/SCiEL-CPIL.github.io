"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import numpy as np

def _primes(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = False
    return np.array([p for p, ok in enumerate(sieve) if ok], dtype=int)

def construct():
    primes = _primes(N)
    num_primes = len(primes)
    
    # Precompute residues for each x for each prime
    # residues[i] is an array of length N+1, residues[i][x] = x % primes[i]
    residues = np.zeros((num_primes, N + 1), dtype=int)
    for i, p in enumerate(primes):
        residues[i] = np.arange(N + 1) % p
        
    # Initialize counts: counts[i][r] = number of active x such that x % p_i == r
    counts = np.zeros((num_primes, 0), dtype=int) # Placeholder, will resize
    # Actually, let's just use a list of arrays or a 2D array with max prime size?
    # Max prime is 5000. So shape (num_primes, max_prime) is wasteful but okay.
    # Better: list of arrays.
    counts = []
    for i, p in enumerate(primes):
        c = np.bincount(residues[i], minlength=p)
        counts.append(c)
        
    active = np.ones(N + 1, dtype=bool)
    current_size = N + 1
    
    # Track which primes are satisfied (have a 0 count in some residue)
    # We need to find the prime with the minimum non-zero count among unsatisfied primes <= current_size
    # Heap of (min_count, prime_idx)
    import heapq
    heap = []
    for i in range(num_primes):
        min_c = np.min(counts[i])
        heapq.heappush(heap, (min_c, i))
        
    while heap:
        cost, p_idx = heapq.heappop(heap)
        p = primes[p_idx]
        
        # Check if this prime is still relevant (p <= current_size)
        if p > current_size:
            continue
            
        # Get current min count
        cur_min = np.min(counts[p_idx])
        
        if cur_min == 0:
            # This prime is already satisfied (a residue class is empty)
            # We don't need to do anything, but we shouldn't push it back if it's satisfied?
            # Actually, if it's satisfied, it stays satisfied.
            continue
            
        # If the cost in heap is not the current min, it's stale
        if cur_min != cost:
            heapq.heappush(heap, (cur_min, p_idx))
            continue
            
        # This is the best prime to fix
        # Find the residue with min count
        r = np.argmin(counts[p_idx])
        
        # Remove all elements x where x % p == r
        # Mask: residues[p_idx] == r
        mask = residues[p_idx] == r
        # Only remove if active
        to_remove_idx = np.where(mask & active)[0]
        
        if len(to_remove_idx) == 0:
            # Should not happen if cost > 0
            continue
            
        # Update active
        active[to_remove_idx] = False
        current_size -= len(to_remove_idx)
        
        # Update counts for all primes
        # For each prime j, decrement count for residue residues[j][x] for x in to_remove_idx
        for j in range(num_primes):
            # Get residues of removed elements for prime j
            rem_residues = residues[j][to_remove_idx]
            # Use bincount to get decrements
            decs = np.bincount(rem_residues, minlength=primes[j])
            counts[j] -= decs
            
        # Push updated min for this prime back? 
        # It might still be the best, or another prime might be better.
        # Let's just push the new min for this prime.
        new_min = np.min(counts[p_idx])
        heapq.heappush(heap, (new_min, p_idx))
        
    return np.where(active)[0].tolist()
# EVOLVE-BLOCK-END
