"""Combined program: NumPy vectorized coverage + Guided Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: return before 120s kill. 110s is safe.
    deadline = start_time + 110.0
    
    # Generate primes up to 1000
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
    
    # Score is capped at 10^5.
    MAX_L = 100000
    coverage = np.zeros(MAX_L + 1, dtype=np.int32)
    
    # Initialize residues randomly
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Initial coverage calculation
    for i, p in enumerate(primes):
        r = residues[i]
        # x % p == r => x = r, r+p, r+2p...
        # In numpy slice coverage[r::p], index r corresponds to x=r.
        # We care about x in 1..MAX_L.
        # If r == 0, indices are 0, p, 2p... We want 1..MAX_L.
        # coverage[0] is index 0. We check coverage[1:] in get_L.
        # So updating coverage[r::p] is correct for x>=r.
        # If r=0, it updates 0, p, 2p... which includes p, 2p... (x>=p>=2).
        # The index 0 (x=0) is updated but ignored in get_L (which starts at 1).
        coverage[r::p] += 1
        
    def get_L():
        # Find first x in 1..MAX_L where coverage[x] < 2
        # coverage[1:] has length MAX_L.
        # np.where returns indices in the slice.
        # If slice index 0 (x=1) is invalid, return 0.
        # If slice index 1 (x=2) is invalid, return 1.
        # So L = first_invalid_index_in_slice.
        invalid = np.where(coverage[1:] < 2)[0]
        if len(invalid) == 0:
            return MAX_L
        else:
            return int(invalid[0])
            
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing parameters
    # Start with a temperature that allows some exploration
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    exp = math.exp
    rand = random.random
    randint = random.randint
    
    steps = 0
    
    # Pre-compute indices for speed? No, n_primes is small (168).
    
    while time.time() < deadline:
        # Decide move type
        # 90% guided, 10% random
        use_guided = rand() < 0.9
        
        if use_guided and current_L < MAX_L:
            x_bad = current_L + 1
            # Find primes that do NOT cover x_bad
            # Condition: residues[i] != x_bad % primes[i]
            candidates = []
            for i in range(n_primes):
                p = primes[i]
                if residues[i] != x_bad % p:
                    candidates.append(i)
            
            if candidates:
                # Pick a random candidate from those that don't cover x_bad
                i = candidates[randint(0, len(candidates) - 1)]
                p = primes[i]
                old_r = residues[i]
                new_r = x_bad % p
                
                # Update coverage
                coverage[old_r::p] -= 1
                coverage[new_r::p] += 1
                
                new_L = get_L()
                
                # Acceptance criteria
                if new_L > current_L:
                    current_L = new_L
                    residues[i] = new_r
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                else:
                    delta = new_L - current_L
                    if delta < 0:
                        # Calculate acceptance probability
                        # Avoid overflow/underflow issues with large negative delta
                        if delta < -100: 
                            prob = 0.0
                        else:
                            prob = exp(delta / T)
                        if rand() < prob:
                            current_L = new_L
                            residues[i] = new_r
                        else:
                            # Revert
                            coverage[new_r::p] -= 1
                            coverage[old_r::p] += 1
                    else:
                        # delta == 0, accept to move around plateau
                        current_L = new_L
                        residues[i] = new_r
            else:
                # All primes cover x_bad? This means coverage[x_bad] >= n_primes >= 2.
                # But get_L said coverage[x_bad] < 2. Contradiction.
                # This case shouldn't happen if logic is correct.
                # Fallback to random move.
                pass
        else:
            # Random move
            i = randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = randint(0, p - 1)
            
            if old_r == new_r:
                continue
            
            coverage[old_r::p] -= 1
            coverage[new_r::p] += 1
            
            new_L = get_L()
            
            if new_L > current_L:
                current_L = new_L
                residues[i] = new_r
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                delta = new_L - current_L
                if delta < 0:
                    if delta < -100:
                        prob = 0.0
                    else:
                        prob = exp(delta / T)
                    if rand() < prob:
                        current_L = new_L
                        residues[i] = new_r
                    else:
                        # Revert
                        coverage[new_r::p] -= 1
                        coverage[old_r::p] += 1
                else:
                    current_L = new_L
                    residues[i] = new_r
        
        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T
            
        steps += 1
        
        # Check time less frequently to reduce overhead
        if steps % 10 == 0 and time.time() >= deadline:
            break
            
    return best_residues
# EVOLVE-BLOCK-END
