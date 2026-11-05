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
    MAX_L = 20000
    
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
        bad_indices = np.where(counts[1:] < 2)[0]
        if len(bad_indices) == 0:
            return MAX_L
        else:
            return bad_indices[0]

    # Greedy Initialization
    # We want to maximize coverage for small integers.
    # Start with residues = 0? Or 1?
    # Let's try a simple greedy: for each prime, pick the residue that maximizes
    # the number of integers in [1, 1000] that become covered (count >= 2) or 
    # minimizes the number of integers with count < 2.
    
    # A better init: random, then hill climb.
    random.seed(42)
    residues = [random.randint(0, p - 1) for p in primes]
    counts = update_counts(residues, counts, primes, MAX_L)
    
    current_L = get_L(counts)
    best_L = current_L
    best_residues = residues[:]
    
    start_time = time.time()
    time_limit = 115.0
    
    # Hill Climbing with Best Improvement
    # We iterate through primes. For each prime, we try all residues and keep the best.
    # To speed up, we can shuffle the order of primes or focus on smaller ones.
    
    # Optimization: Pre-compute indices for each prime?
    # No, just use slicing.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Iterate through primes in a random order or increasing order?
        # Increasing order might be better for local structure.
        # Let's shuffle to escape local minima? No, best improvement is deterministic.
        # We can use random restarts if stuck.
        
        # Try to improve by changing one prime at a time, choosing the best change.
        # We'll do multiple passes.
        
        # To avoid getting stuck in a local minimum where no single prime change helps,
        # we might need to consider pairs or random restarts.
        # But first, let's see how far single-prime best improvement gets us.
        
        # Randomly select a prime to modify
        # Bias towards smaller primes?
        # Smaller primes have more steps in the array, so changing them has more impact.
        
        # Let's try a greedy pass: for each prime, find the best residue.
        # This is O(N_primes * avg_p * cost_update).
        # cost_update is O(MAX_L/p).
        # Total cost approx sum(p * MAX_L/p) = N_primes * MAX_L.
        # 168 * 20000 = 3.36M operations. This is very fast in Python if vectorized.
        
        for i in range(n_primes):
            if time.time() - start_time > time_limit:
                break
                
            p = primes[i]
            old_r = residues[i]
            
            # Remove old contribution
            start_old = old_r
            if start_old == 0:
                start_old = p
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
                
            best_new_r = old_r
            best_new_L = current_L
            best_counts = None # We don't want to copy counts array, it's expensive.
            
            # Try all residues
            # We can optimize by only checking a subset if p is large?
            # For p > 100, there are >100 residues.
            # Let's just try all, it's fast enough.
            
            for new_r in range(p):
                if new_r == old_r:
                    continue
                
                start_new = new_r
                if start_new == 0:
                    start_new = p
                if start_new <= MAX_L:
                    counts[start_new::p] += 1
                
                # Evaluate L
                # This is the bottleneck. get_L is O(MAX_L) with np.where.
                # 168 * 1000 * 20000 = 3.36 * 10^9 operations? No.
                # Inner loop is p. Outer is n_primes.
                # Total evaluations: sum(p for p in primes) ~ 76127.
                # Each evaluation is np.where on 20000 array.
                # 76127 * 20000 = 1.5 * 10^9. This might be too slow in Python loop overhead.
                
                # We need to speed up get_L.
                # Instead of np.where every time, we can maintain a count of "bad" spots?
                # Or just accept it might be slow and reduce MAX_L or use a heuristic.
                
                # Let's optimize get_L:
                # We only care if L increases.
                # If current_L is 1000, we only need to check if 1001 is covered.
                # Actually, L is the *first* uncovered spot.
                # If we change a residue, L can only increase or decrease.
                # If it increases, the new L is > current_L.
                # We can check counts[current_L] first. If it's >= 2, we might have extended L.
                # Then we need to find the new first bad spot.
                
                # For now, let's just use a faster check.
                # If counts[current_L] < 2, L hasn't increased (or decreased).
                # Actually, if we remove a cover, L might decrease.
                # If we add a cover, L might increase.
                
                # Let's just run get_L but optimize the numpy call.
                # np.where is C-speed, so it should be okay.
                # 76k iterations * ~20us per np.where = 1.5s. This is acceptable.
                
                new_L = get_L(counts)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_new_r = new_r
                    # Don't copy counts, we will revert and re-apply later
                # Revert this trial
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
            
            # Apply best_new_r
            if best_new_r != old_r:
                start_best = best_new_r
                if start_best == 0:
                    start_best = p
                if start_best <= MAX_L:
                    counts[start_best::p] += 1
                residues[i] = best_new_r
                current_L = best_new_L
                improved = True
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
            else:
                # Revert to old_r (it was already removed, so add it back? No, we removed old_r at start)
                # Wait, logic:
                # 1. Remove old_r
                # 2. Try new_r, revert new_r
                # 3. If best_new_r == old_r, we need to add old_r back.
                # If best_new_r != old_r, we add best_new_r.
                # So in both cases, we add the chosen residue.
                # But if best_new_r == old_r, we just added old_r back?
                # Yes, the code above `if best_new_r != old_r` handles the change.
                # If it doesn't change, we must restore old_r.
                # My logic above:
                # - Removed old_r.
                # - Tried others, reverted them.
                # - If best is old_r, we didn't add it back yet.
                # - If best is new_r, we add new_r.
                # So:
                if best_new_r == old_r:
                     start_old = old_r
                     if start_old == 0: start_old = p
                     if start_old <= MAX_L: counts[start_old::p] += 1

        if not improved:
            # Stuck in local minimum.
            # Try a random perturbation: change 2-3 random primes to random values.
            if best_L >= MAX_L:
                break # Can't do better than MAX_L with this window
            
            # Perturb
            for _ in range(3):
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                old_r = residues[idx]
                
                start_old = old_r
                if start_old == 0: start_old = p
                if start_old <= MAX_L: counts[start_old::p] -= 1
                
                new_r = random.randint(0, p - 1)
                residues[idx] = new_r
                
                start_new = new_r
                if start_new == 0: start_new = p
                if start_new <= MAX_L: counts[start_new::p] += 1
            
            current_L = get_L(counts)
            # Don't update best_L here, let the next loop find improvements.
            # If this perturbation made it worse, the next "best improvement" pass will try to fix it.
            # But if it's stuck, we might need to accept a worse state temporarily?
            # Hill climbing doesn't do that. We need Simulated Annealing or Tabu Search.
            # Let's switch to a simpler SA if Hill Climbing stalls?
            # For now, let's just restart with a different seed if we stall too long?
            # Or just continue.
            
            # To avoid infinite loop of no improvement, we can break and restart with new random init.
            # But we have limited time.
            # Let's just break the inner while and restart with a new random initialization.
            break

    return best_residues

# EVOLVE-BLOCK-END
