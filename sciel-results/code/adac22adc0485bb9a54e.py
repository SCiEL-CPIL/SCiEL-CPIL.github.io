import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Generate Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(1000)
    n_primes = len(primes)
    MAX_L = 1000  # Search up to 1000, cap score at 10^5 but we are looking for L>=1000
    
    # 2. Initialization
    # Heuristic: Start with residues that cover small integers well.
    # r_p = 0 is bad for x=1.
    # Let's try r_p = 1 % p for all p? Then x=1 is covered by all.
    # x=2 covered by p where 2%p==r_p.
    # Let's use the seed's idea but slightly modified or random.
    # Seed: ((i + 2) // 2) % p
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # 3. Coverage Structure
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    
    def update_counts(p_idx, new_r):
        p = primes[p_idx]
        old_r = residues[p_idx]
        if old_r == new_r:
            return
        
        # Calculate indices in [1, MAX_L]
        # x = k*p + r, x > 0
        # First x > 0:
        # if r == 0, x = p, 2p, ...
        # if r > 0, x = r, r+p, ... (if r <= MAX_L)
        
        # Vectorized approach might be slower for small p due to array creation overhead?
        # Looping might be faster for small p?
        # Let's use numpy arange.
        
        # Old values
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        if start_old <= MAX_L:
            old_vals = np.arange(start_old, MAX_L + 1, p)
            counts[old_vals] -= 1
            
        # New values
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        if start_new <= MAX_L:
            new_vals = np.arange(start_new, MAX_L + 1, p)
            counts[new_vals] += 1
            
        residues[p_idx] = new_r

    # Initialize counts
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= MAX_L:
            vals = np.arange(start, MAX_L + 1, p)
            counts[vals] += 1

    def get_L():
        # Find first index i in 1..MAX_L where counts[i] < 2
        # counts[0] is unused
        under_covered = np.where(counts[1:] < 2)[0]
        if len(under_covered) == 0:
            return MAX_L
        return under_covered[0] + 1

    best_L = get_L()
    best_residues = residues.copy()
    current_L = best_L
    
    # 4. Optimization Loop
    stuck_counter = 0
    
    while time.time() < deadline:
        # 1. Targeted Move
        if current_L < MAX_L:
            gap = current_L + 1
            gap_count = counts[gap]
            
            # We want to increase count at gap.
            # If gap_count < 2, we need to add coverage.
            # Primes that do NOT cover gap can be switched to cover it.
            # Primes that DO cover gap cannot be switched to cover it (already do).
            
            # Candidate primes: those where residues[i] != gap % p
            # We pick one at random or systematically.
            
            # Try a few random candidates
            attempts = 0
            max_attempts = 20 # Limit per iteration to keep loop fast
            
            while attempts < max_attempts and time.time() < deadline:
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                needed_r = gap % p
                current_r = residues[i]
                
                if current_r == needed_r:
                    attempts += 1
                    continue
                
                # Try changing i to needed_r
                old_r = current_r
                update_counts(i, needed_r)
                new_L = get_L()
                
                if new_L > current_L:
                    current_L = new_L
                    stuck_counter = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                    # Don't revert
                    break # Accept and restart outer loop logic
                else:
                    # Revert
                    update_counts(i, old_r)
                    attempts += 1
            else:
                # No improvement found in this batch
                stuck_counter += 1
                
                # If stuck for a while, do a random kick
                if stuck_counter > 50:
                    # Randomize 5 residues
                    for _ in range(5):
                        i = random.randint(0, n_primes - 1)
                        p = primes[i]
                        new_r = random.randint(0, p - 1)
                        update_counts(i, new_r)
                    
                    new_L = get_L()
                    if new_L < current_L:
                        # This is a "kick", so we accept even if worse, to escape local min
                        # But we must update current_L
                        current_L = new_L
                    else:
                        if new_L > best_L:
                            best_L = new_L
                            best_residues = residues.copy()
                        current_L = new_L
                    stuck_counter = 0
        else:
            # Reached MAX_L, we can stop or keep trying to ensure no errors
            # Since score is capped, breaking is fine if we are confident.
            # But maybe L can go higher? MAX_L is 1000.
            # If we reach 1000, we break.
            break

    return best_residues
