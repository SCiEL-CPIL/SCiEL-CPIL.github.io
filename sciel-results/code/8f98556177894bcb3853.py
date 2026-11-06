import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    # Use 110s deadline to safely return before 120s kill
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
    
    # Score is capped at 10^5. We don't need to track L beyond that for the score.
    # Using MAX_L = 100000 reduces memory and scan time.
    MAX_L = 100000
    coverage = np.zeros(MAX_L + 1, dtype=np.int32)
    
    best_L = 0
    best_residues = [0] * n_primes
    
    # Pre-compute modulo values for each prime for each position? 
    # That would be 168 * 100000 = 16.8M ints, ~67MB. 
    # Alternatively, compute x % p on the fly. Python modulo is fast enough for 168 primes.
    
    def get_L():
        # Find first x in 1..MAX_L where coverage[x] < 2
        invalid = np.where(coverage[1:] < 2)[0]
        if len(invalid) == 0:
            return MAX_L
        else:
            return int(invalid[0])
            
    def update_coverage(residues, coverage):
        """Rebuild coverage from scratch. Used at start of restart."""
        coverage[:] = 0
        for i, p in enumerate(primes):
            r = residues[i]
            coverage[r::p] += 1

    # Multi-restart strategy
    random.seed(42)
    
    while time.time() < deadline - 5.0: # Leave 5s margin for final cleanup/return
        
        # Initialization: 50% random, 50% perturb best
        if best_L > 0 and random.random() < 0.5:
            residues = best_residues[:]
            # Perturb 10-20% of residues to escape local minima
            num_perturb = max(1, n_primes // 10)
            for _ in range(num_perturb):
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                residues[idx] = random.randint(0, p - 1)
        else:
            residues = [random.randint(0, p-1) for p in primes]
        
        update_coverage(residues, coverage)
        current_L = get_L()
        
        if current_L > best_L:
            best_L = current_L
            best_residues = residues[:]
            if best_L >= MAX_L:
                break

        # Phase 1: Greedy Repair
        # Try to extend L by fixing the first gap
        stuck_count = 0
        repair_steps = 0
        max_repair_steps = 5000 # Limit to avoid infinite loops in one restart
        
        while current_L < MAX_L and repair_steps < max_repair_steps and time.time() < deadline - 2.0:
            x_bad = current_L + 1
            
            # Find primes that do NOT cover x_bad
            # r_p != x_bad % p
            candidates = []
            for i in range(n_primes):
                p = primes[i]
                if residues[i] != x_bad % p:
                    candidates.append(i)
            
            if not candidates:
                # All primes cover x_bad, but coverage < 2? Impossible if all 168 cover.
                # This implies a bug in logic or coverage tracking.
                # Break to avoid infinite loop.
                break
            
            # Try up to 10 random candidates from the list
            random.shuffle(candidates)
            improved = False
            
            for i in candidates[:10]:
                p = primes[i]
                old_r = residues[i]
                new_r = x_bad % p
                
                # Update coverage
                coverage[old_r::p] -= 1
                coverage[new_r::p] += 1
                
                new_L = get_L()
                
                if new_L > current_L:
                    current_L = new_L
                    residues[i] = new_r
                    improved = True
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
                    break # Restart gap finding with new L
                else:
                    # Revert
                    coverage[new_r::p] -= 1
                    coverage[old_r::p] += 1
            
            if not improved:
                # Stuck. Try Simulated Annealing escape
                stuck_count += 1
                if stuck_count > 5:
                    break
                # Perform a few SA steps to escape
                T = 5.0
                for _ in range(50):
                    if time.time() > deadline - 2.0:
                        break
                    i = random.randint(0, n_primes - 1)
                    p = primes[i]
                    old_r = residues[i]
                    new_r = random.randint(0, p - 1)
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
                            best_residues = residues[:]
                        stuck_count = 0 # Reset stuck counter
                        break
                    else:
                        delta = new_L - current_L
                        if delta < 0:
                            prob = math.exp(delta / T)
                            if random.random() < prob:
                                current_L = new_L
                                residues[i] = new_r
                            else:
                                # Revert
                                coverage[new_r::p] -= 1
                                coverage[old_r::p] += 1
                        else:
                            # delta == 0, accept to wander
                            current_L = new_L
                            residues[i] = new_r
                
                # If we improved during SA, we will loop back to greedy repair
                # If not, we might still be stuck
                if stuck_count > 10:
                    break
            else:
                stuck_count = 0 # Reset if we made progress
                repair_steps += 1

        # If we reached a high L, we can stop early to save time for other restarts?
        # No, more restarts might find better. But if best_L is very high, 
        # the probability of improving is low. 
        # However, the task asks to maximize. 
        # If best_L > 5000, we might switch to a pure SA phase for the remaining time 
        # starting from best_residues to polish.
        
        if best_L > 2000:
             # Polish phase: SA from best_residues
             residues = best_residues[:]
             update_coverage(residues, coverage)
             current_L = get_L()
             
             T = 10.0
             cooling = 0.999
             min_T = 0.5
             
             while time.time() < deadline - 2.0:
                 i = random.randint(0, n_primes - 1)
                 p = primes[i]
                 old_r = residues[i]
                 new_r = random.randint(0, p - 1)
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
                         best_residues = residues[:]
                 else:
                     delta = new_L - current_L
                     if delta < 0:
                         prob = math.exp(delta / T)
                         if random.random() < prob:
                             current_L = new_L
                             residues[i] = new_r
                         else:
                             coverage[new_r::p] -= 1
                             coverage[old_r::p] += 1
                     else:
                         current_L = new_L
                         residues[i] = new_r
                         
                 T *= cooling
                 if T < min_T:
                     T = min_T

    return [int(r) for r in best_residues]
