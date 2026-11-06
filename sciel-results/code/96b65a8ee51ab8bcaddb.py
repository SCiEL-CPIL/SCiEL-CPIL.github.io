"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys
import numpy as np

def construct():
    # Deadline guard: 150s limit, use 148s for safety
    start_time = time.time()
    deadline = start_time + 148.0
    
    # Generate primes up to 1000
    def get_primes():
        sieve = [True] * 1001
        sieve[0] = sieve[1] = False
        for i in range(2, int(1000**0.5) + 1):
            if sieve[i]:
                sieve[i*i:1001:i] = [False] * len(sieve[i*i:1001:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes()
    n_primes = len(primes)
    
    # Precompute for each prime index i, the list of positions in 1..MAX_L that it covers for each residue
    # This allows O(1) updates if we use numpy or efficient loops.
    # But for dynamic L, we might just compute on the fly or precompute up to a large limit.
    # Let's precompute masks for a large L, say 5000, to allow fast extension.
    MAX_L_PRECOMP = 5000
    
    # coverage[p_idx][r] is a boolean array of size MAX_L_PRECOMP indicating which x are covered
    # Actually, storing 168 * p * 5000 booleans is too much memory (168 * 1000 * 5000 ~ 840MB).
    # Instead, we update coverage incrementally.
    
    # State:
    # residues: list of 168 ints
    # coverage: numpy array of size MAX_L_PRECOMP + 1, counting how many primes cover each x
    
    def calculate_coverage(residues, L_limit):
        cov = np.zeros(L_limit + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            # x such that x % p == r, x in 1..L_limit
            # first x = r if r > 0 else p
            start = r if r > 0 else p
            if start <= L_limit:
                cov[start:L_limit+1:p] += 1
        return cov

    def get_L(coverage, L_limit):
        # Find largest L such that coverage[1..L] >= 2
        # We can use numpy for speed
        valid = coverage[1:L_limit+1] >= 2
        if not np.any(valid):
            return 0
        # Find the first False
        # If all True, L = L_limit
        false_indices = np.where(~valid)[0]
        if len(false_indices) == 0:
            return L_limit
        return false_indices[0] # This is the 1-based index of the first failure, so L is that index - 1? 
        # Wait. valid[0] corresponds to x=1.
        # If valid[0] is False, L=0.
        # If valid[0] is True, valid[1] is False, L=1.
        # So L = index of first False.
        
    # Strategy: Greedy Interval Extension
    # 1. Start with L=0, empty assignment (or random).
    # 2. Try to extend to L+1.
    # 3. If stuck, use local search to fix violations in 1..L+1.
    # 4. If successful, increment L and repeat.
    
    # Better: 
    # Initialize residues randomly.
    # Calculate L.
    # While time remains:
    #   Identify the "bottleneck" x = L+1 (or the first x with cov < 2).
    #   Try to modify residues to increase coverage of x and nearby numbers.
    #   Use a "repair" heuristic: find a prime p that currently does NOT cover x, 
    #   and changing its residue to x%p would increase coverage of x without decreasing coverage of 1..L too much.
    
    # Let's implement a robust Local Search that optimizes for the longest prefix.
    
    def evaluate(residues):
        # Returns L
        # We only care about 1..some_limit
        # To be fast, we can stop early if we find a hole.
        # But numpy is fast enough for 5000.
        cov = calculate_coverage(residues, MAX_L_PRECOMP)
        return get_L(cov, MAX_L_PRECOMP)

    # Initialize
    random.seed(42)
    best_residues = [random.randint(0, p-1) for p in primes]
    best_L = evaluate(best_residues)
    
    # Local Search Loop
    # We will use a "Hill Climbing" with restarts or a "Simulated Annealing" but 
    # with a smarter move set: 
    # Move: Pick a prime p. Change r_p.
    # Objective: Maximize L.
    # Tie-breaker: Maximize sum of coverage in 1..L (to make it more robust).
    
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    current_residues = best_residues[:]
    current_L = best_L
    
    # Precompute for each prime p, the list of x in 1..MAX_L_PRECOMP for each residue?
    # No, we update incrementally.
    
    cov = calculate_coverage(current_residues, MAX_L_PRECOMP)
    
    def get_score(cov):
        L = get_L(cov, MAX_L_PRECOMP)
        # Secondary objective: sum of coverage in 1..L
        if L == 0:
            return 0, 0
        # Sum of coverage for 1..L
        # We want high coverage to be "safe"
        sec_score = np.sum(cov[1:L+1])
        return L, sec_score
    
    cur_L, cur_sec = get_score(cov)
    
    while time.time() < deadline:
        # Pick a random prime index
        idx = random.randint(0, n_primes - 1)
        p = primes[idx]
        old_r = current_residues[idx]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        # Update coverage
        # Remove old_r
        start_old = old_r if old_r > 0 else p
        if start_old <= MAX_L_PRECOMP:
            cov[start_old:MAX_L_PRECOMP+1:p] -= 1
            
        # Add new_r
        start_new = new_r if new_r > 0 else p
        if start_new <= MAX_L_PRECOMP:
            cov[start_new:MAX_L_PRECOMP+1:p] += 1
            
        current_residues[idx] = new_r
        
        new_L, new_sec = get_score(cov)
        
        # Acceptance
        # If new_L > cur_L, accept.
        # If new_L == cur_L and new_sec > cur_sec, accept.
        # If new_L < cur_L, accept with prob exp(-delta/T)
        
        if new_L > cur_L:
            # Improve
            cur_L = new_L
            cur_sec = new_sec
            if cur_L > best_L:
                best_L = cur_L
                best_residues = current_residues[:]
        elif new_L == cur_L:
            if new_sec >= cur_sec:
                cur_sec = new_sec
                # Keep best if sec score is better? No, best is defined by L.
                # But we track best_L separately.
            else:
                # Reject or accept with prob?
                # Since L is same, we can be more aggressive.
                delta = cur_sec - new_sec
                prob = math.exp(-delta / (T * 10)) # Scale down penalty for sec score
                if random.random() < prob:
                    cur_sec = new_sec
                else:
                    # Revert
                    cov[start_new:MAX_L_PRECOMP+1:p] -= 1
                    cov[start_old:MAX_L_PRECOMP+1:p] += 1
                    current_residues[idx] = old_r
        else:
            # new_L < cur_L
            delta = cur_L - new_L
            prob = math.exp(-delta / T)
            if random.random() < prob:
                cur_L = new_L
                cur_sec = new_sec
            else:
                # Revert
                cov[start_new:MAX_L_PRECOMP+1:p] -= 1
                cov[start_old:MAX_L_PRECOMP+1:p] += 1
                current_residues[idx] = old_r
        
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: Restart from best
            if random.random() < 0.1:
                current_residues = best_residues[:]
                cov = calculate_coverage(current_residues, MAX_L_PRECOMP)
                cur_L, cur_sec = get_score(cov)
                
    return best_residues

# EVOLVE-BLOCK-END
