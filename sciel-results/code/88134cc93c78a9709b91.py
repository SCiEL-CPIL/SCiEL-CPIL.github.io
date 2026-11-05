"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import random
import time

def construct():
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
    
    # We will optimize all primes
    # Initialize residues
    # Heuristic: Start with r_p = 1 for all, or random
    residues = [1] * n_primes
    
    # Max L to check during search
    # The problem asks for L >= 1000. Let's aim a bit higher to be safe, say 5000.
    # If we reach 5000, the score is at least 1000.
    MAX_L = 5000
    
    # Initialize counts array
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    
    def update_counts(residues, counts, primes, MAX_L):
        counts[:] = 0
        for i, p in enumerate(primes):
            r = residues[i]
            start = r
            if start == 0:
                start = p
            if start <= MAX_L:
                counts[start::p] += 1
        return counts

    def get_L(counts):
        # Find first x in 1..MAX_L such that counts[x] < 2
        # Using numpy for speed
        # We want the first index where counts < 2.
        # If no such index exists, return MAX_L.
        # Note: counts[0] is not checked in problem statement (1..L), but we initialized counts for 1..MAX_L.
        # Actually, the problem says "every integer 1..L".
        # So we check indices 1 to MAX_L.
        
        # Create a boolean mask where count < 2
        # We only care about indices 1..MAX_L
        # If all are >= 2, argmax on an all-False array returns 0, which is wrong.
        # So check if any are < 2.
        
        # Efficient way:
        # np.where(counts[1:] < 2)[0]
        # If empty, return MAX_L. Else return first index + 1 (since we sliced from 1, index 0 in slice is index 1 in counts)
        # Wait, if slice is counts[1:], index 0 in result corresponds to counts[1].
        # So L = index_in_slice + 1 - 1 = index_in_slice?
        # Let's trace:
        # If counts[1] < 2, slice index 0 is True. Result index 0. L should be 0 (since 1 is not covered).
        # If counts[1] >= 2, counts[2] < 2, slice index 1 is True. Result index 1. L should be 1.
        # So L = first_true_index_in_slice.
        
        bad_indices = np.where(counts[1:] < 2)[0]
        if len(bad_indices) == 0:
            return MAX_L
        else:
            return bad_indices[0]

    # Initial greedy setup: try to set residues to maximize immediate coverage?
    # Just use random or 1 for now, then anneal.
    random.seed(42)
    
    # Better initialization: Random
    residues = [random.randint(0, p - 1) for p in primes]
    counts = update_counts(residues, counts, primes, MAX_L)
    
    current_L = get_L(counts)
    best_L = current_L
    best_residues = residues[:]
    
    # Simulated Annealing Parameters
    # We will run multiple iterations (restarts) if time permits
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Parameters for each annealing run
    T_start = 5.0
    T_min = 0.001
    cooling_rate = 0.9995
    max_iter_per_run = 50000
    
    while time.time() - start_time < time_limit:
        T = T_start
        residues = best_residues[:]
        counts = update_counts(residues, counts, primes, MAX_L)
        current_L = get_L(counts)
        
        # If we start from best, current_L is best_L
        # We will try to improve from here
        
        for i in range(max_iter_per_run):
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random prime index
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            
            # Pick a random new residue
            new_r = random.randint(0, p - 1)
            old_r = residues[idx]
            
            if new_r == old_r:
                continue
            
            # Update counts: Remove old_r coverage
            start_old = old_r
            if start_old == 0:
                start_old = p
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
            
            # Update counts: Add new_r coverage
            start_new = new_r
            if start_new == 0:
                start_new = p
            if start_new <= MAX_L:
                counts[start_new::p] += 1
            
            # Evaluate new L
            new_L = get_L(counts)
            
            # Acceptance Criteria
            if new_L >= current_L:
                # Accept improvement or equal
                current_L = new_L
                residues[idx] = new_r
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues[:]
            else:
                # Worse solution
                # Accept with probability exp(-delta / T)
                delta = current_L - new_L
                if delta < T:
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                
                if random.random() < prob:
                    current_L = new_L
                    residues[idx] = new_r
                else:
                    # Revert
                    if start_new <= MAX_L:
                        counts[start_new::p] -= 1
                    if start_old <= MAX_L:
                        counts[start_old::p] += 1
                    # residues[idx] remains old_r
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                break

    return best_residues

# EVOLVE-BLOCK-END
