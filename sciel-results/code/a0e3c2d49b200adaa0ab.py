"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

# Precompute primes and masks
def _init_problem():
    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # Precompute residue masks for all primes
    # masks[p_index][r] is a numpy array of booleans/ints
    # To save memory and time, we can store them as arrays of 0/1
    # Number of primes ~ 669
    # Total entries ~ sum(p) ~ 30000
    # Each entry is N+1 bytes ~ 5KB
    # Total ~ 150MB. This might be high for some environments, but usually okay.
    # Alternatively, compute on the fly. 
    # Let's compute on the fly to be safe on memory, but cache the first K.
    
    return primes

PRIMES = _init_problem()
K_OPTIMIZE = 15 # Optimize first 15 primes

# Precompute masks for the first K primes to speed up the "fixed" part?
# Actually, the evaluation loop is the bottleneck.
# Let's precompute masks for ALL primes if memory allows.
# 669 primes * avg 20 residues * 5000 bytes = 66.9 MB. Acceptable.
MASKS = {}
for p in PRIMES:
    masks_for_p = []
    for r in range(p):
        # Create mask where x % p == r
        # Using np.arange
        mask = (np.arange(N + 1) % p == r).astype(np.uint8)
        masks_for_p.append(mask)
    MASKS[p] = masks_for_p

ARANGE = np.arange(N + 1)

def _evaluate(residues):
    """
    residues: dict {prime: residue} for the first K primes.
    Returns the size of the admissible set.
    """
    current_mask = np.ones(N + 1, dtype=np.uint8)
    
    # Apply fixed choices for first K primes
    for p in PRIMES[:K_OPTIMIZE]:
        r = residues[p]
        current_mask &= ~MASKS[p][r]
        
    # Apply greedy choices for the rest
    for p in PRIMES[K_OPTIMIZE:]:
        if len(np.nonzero(current_mask)[0]) < p:
            # If fewer points than p, we can't remove a full class? 
            # Actually we must remove a class. If min count is 0, we remove that class (removes 0 points).
            # If min count > 0, we remove one.
            # The greedy rule "remove min count class" handles this.
            pass
            
        # Compute counts of 1s in each residue class
        # We need to count how many active points fall into each residue
        # residues_mod_p = ARANGE % p
        # counts = np.bincount(residues_mod_p, weights=current_mask, minlength=p)
        
        # Optimization: If current_mask is sparse, this is still O(N).
        # N=5000 is small.
        counts = np.bincount(ARANGE % p, weights=current_mask, minlength=p)
        
        min_count = np.min(counts)
        # Find the first index with min count
        # If min_count == 0, removing it costs 0.
        # If min_count > 0, removing it costs min_count.
        # We want to minimize the loss, so we pick min count.
        r = np.argmin(counts)
        
        current_mask &= ~MASKS[p][r]
        
    return int(np.sum(current_mask))

def construct():
    """
    Construct the largest admissible tuple within [0, N].
    Uses Simulated Annealing on the residue choices for the first K primes.
    """
    start_time = time.time()
    time_limit = 250 # Leave 50s buffer
    
    # Initialize random state
    random.seed(42)
    
    # Initial residues: Random
    current_residues = {p: random.randint(0, p - 1) for p in PRIMES[:K_OPTIMIZE]}
    current_score = _evaluate(current_residues)
    
    best_residues = current_residues.copy()
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    while T > T_min and time.time() - start_time < time_limit:
        # Pick a random prime to change
        p = random.choice(PRIMES[:K_OPTIMIZE])
        
        # Try all other residues for this prime
        best_new_r = current_residues[p]
        best_new_score = current_score
        
        for r in range(p):
            if r == current_residues[p]:
                continue
            # Create new residues dict
            # To avoid deep copying dict, we can modify in place and revert, or just pass a modified view
            # Since _evaluate takes a dict, let's just swap and revert
            old_r = current_residues[p]
            current_residues[p] = r
            new_score = _evaluate(current_residues)
            current_residues[p] = old_r
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_r = r
                
        # Acceptance criterion
        if best_new_score >= current_score:
            # Always accept improvement
            current_residues[p] = best_new_r
            current_score = best_new_score
        else:
            # Accept with probability exp((new - old) / T)
            delta = best_new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_residues[p] = best_new_r
                    current_score = best_new_score
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_residues = current_residues.copy()
            
        T *= cooling

    # Reconstruct the set from best_residues
    # We need to return the list of integers.
    # We run the greedy completion one last time to get the exact set.
    
    final_mask = np.ones(N + 1, dtype=np.uint8)
    
    # Apply best residues for first K
    for p in PRIMES[:K_OPTIMIZE]:
        r = best_residues[p]
        final_mask &= ~MASKS[p][r]
        
    # Apply greedy for rest
    for p in PRIMES[K_OPTIMIZE:]:
        counts = np.bincount(ARANGE % p, weights=final_mask, minlength=p)
        r = np.argmin(counts)
        final_mask &= ~MASKS[p][r]
        
    # Extract indices
    points = np.nonzero(final_mask)[0]
    return sorted(points.tolist())

# EVOLVE-BLOCK-END
