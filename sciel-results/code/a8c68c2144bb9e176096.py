"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return below_two[0]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    # Initialize with the seed strategy: r_i = ceil((i+2)/2) mod p_i
    # This ensures a baseline of L=84
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Precompute indices for fast updates
    # For each prime i, store the start index and step
    # We will maintain a count array and update it as we change residues
    
    max_check = 100000
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    # Initialize counts
    for i, (r, p) in enumerate(zip(residues, primes)):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
            
    current_L = _score(residues, primes, max_check)
    
    # Local search: Hill climbing
    # We try to extend the prefix by finding the first gap and adjusting residues
    # to cover it, while not breaking coverage elsewhere in the prefix.
    
    # Optimization: Instead of full rescore, we can track the "critical" zone.
    # However, full rescore is O(N * max_check/p) which is fast enough for 168 primes.
    # Let's use a more efficient update:
    # When changing residue r_old to r_new for prime p:
    # 1. Subtract 1 from counts at positions congruent to r_old mod p
    # 2. Add 1 to counts at positions congruent to r_new mod p
    # 3. Check if the new configuration has a longer prefix.
    
    # To be efficient, we only care about the prefix up to some limit.
    # Let's set a working limit, say 2000, initially, and expand if we get stuck.
    # Actually, the score is defined up to 10^5, but the prefix breaks early.
    # We can just check the first failure point.
    
    # Let's define a function to get the first failure point from counts
    def get_L(counts):
        # Find first index i in 1..max_check where counts[i] < 2
        # We can use np.where on the slice 1:max_check+1
        # But we want to stop early.
        # Since max_check is 10^5, np.where is fast (C implementation).
        below = np.where(counts[1:max_check+1] < 2)[0]
        if len(below) == 0:
            return max_check
        return below[0]
        
    current_L = get_L(counts)
    
    # List of indices to try. We might prioritize primes that cover the current gap.
    # But simple random or sequential scan works.
    # Let's do a few passes of sequential scan.
    
    improved = True
    passes = 0
    max_passes = 500
    
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        
        # Shuffle order? Or sequential? Sequential might be faster to implement.
        # Let's try sequential first.
        for i in range(n_primes):
            if time.time() > deadline:
                break
            
            p = primes[i]
            old_r = residues[i]
            
            # Generate candidate new residues
            # We can try all residues? p can be up to 997. 
            # 168 * 997 * (cost of update) might be too slow if done naively for every step.
            # Cost of update: ~ 100000/p operations. Sum over p is 100000 * sum(1/p) ~ 100000 * 3 = 300k ops.
            # 168 * 300k = 50M ops per full pass. In Python this is too slow.
            # We need a smarter candidate selection or vectorization.
            
            # Alternative: Only try residues that cover the current first gap?
            # Let gap = current_L + 1.
            # If we can cover gap, L might increase.
            # Residues r such that gap ≡ r (mod p) is just one specific r: r = gap % p.
            # So for each prime, there is at most ONE residue that covers the current gap.
            # If we change the residue to that one, we might fill the gap.
            # But we might break coverage at other points < gap? No, we are extending.
            # We might break coverage at points > gap? Yes.
            # So the strategy:
            # 1. Find current L.
            # 2. Gap = L + 1.
            # 3. For each prime i, calculate r_needed = Gap % p_i.
            # 4. If r_needed != residues[i], check if changing to r_needed improves L.
            
            # This reduces candidates from p to 1 per prime!
            
            r_new = (current_L + 1) % p
            
            if r_new == old_r:
                continue
                
            # Apply change
            start_old = old_r if old_r > 0 else p
            start_new = r_new if r_new > 0 else p
            
            # Subtract old
            if start_old <= max_check:
                counts[start_old::p] -= 1
            # Add new
            if start_new <= max_check:
                counts[start_new::p] += 1
                
            # Check new L
            new_L = get_L(counts)
            
            if new_L > current_L:
                # Accept
                residues[i] = r_new
                current_L = new_L
                improved = True
            else:
                # Revert
                if start_old <= max_check:
                    counts[start_old::p] += 1
                if start_new <= max_check:
                    counts[start_new::p] -= 1
    
    return residues
# EVOLVE-BLOCK-END
