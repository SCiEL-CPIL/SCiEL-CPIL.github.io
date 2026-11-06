import time
import random
import math
from fractions import Fraction
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    def _primes_upto(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    all_primes = _primes_upto(1000)
    # Use primes >= 13 to maximize cardinality within budget
    selected_primes = [p for p in all_primes if p >= 13]
    N = len(selected_primes)
    
    # Verify budget exactly
    cost = sum(Fraction(1, p) for p in selected_primes)
    if cost > Fraction(1, 1):
        # Remove smallest primes (largest reciprocals) until fits
        # Sort by p ascending to remove smallest first
        selected_primes.sort()
        while cost > Fraction(1, 1) and selected_primes:
            p_remove = selected_primes[0]
            cost -= Fraction(1, p_remove)
            selected_primes.pop(0)
            N -= 1

    MAX_L = 100000
    counts = np.zeros(MAX_L + 2, dtype=np.int32)
    
    # Initialize with random residues
    residues = [random.randint(0, p - 1) for p in selected_primes]
    
    # Initialize counts
    for i in range(N):
        p = selected_primes[i]
        r = residues[i]
        start = r if r != 0 else p
        if start <= MAX_L:
            counts[start:MAX_L+1:p] += 1

    def calc_L():
        # Find first index i in 1..MAX_L where counts[i] == 0
        # L is the count of consecutive covered integers starting from 1
        # If counts[1] == 0, L = 0
        # If counts[1] > 0 and counts[2] == 0, L = 1
        zero_mask = counts[1:MAX_L+1] == 0
        if not np.any(zero_mask):
            return MAX_L
        first_zero_idx = np.argmax(zero_mask)
        # first_zero_idx is 0-based in the slice [1..MAX_L]
        # The actual index is first_zero_idx + 1
        # L is the index before the first zero, so L = first_zero_idx
        return first_zero_idx

    def apply_change(i, old_r, new_r):
        p = selected_primes[i]
        # Decrement old
        start_old = old_r if old_r != 0 else p
        if start_old <= MAX_L:
            counts[start_old:MAX_L+1:p] -= 1
        # Increment new
        start_new = new_r if new_r != 0 else p
        if start_new <= MAX_L:
            counts[start_new:MAX_L+1:p] += 1

    def greedy_extend(current_L):
        extended = True
        L_local = current_L
        # Limit iterations to prevent hanging if stuck in a cycle or very slow progress
        max_iters = 5000 
        iters = 0
        while extended and iters < max_iters and time.time() < deadline:
            extended = False
            iters += 1
            target_val = L_local + 1
            if target_val > MAX_L:
                break
                
            # Try to find a prime that can cover target_val safely
            # Shuffle to avoid bias
            order = list(range(N))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    return L_local
                    
                p = selected_primes[i]
                r_old = residues[i]
                r_new = target_val % p
                
                if r_old == r_new:
                    continue
                
                # Check safety:
                # All n in [1, L_local] covered by old_r must have count > 1
                start_chk = r_old if r_old != 0 else p
                if start_chk > L_local:
                    # Old residue doesn't cover any n in [1, L_local]
                    safe = True
                else:
                    # Check counts for n = start_chk, start_chk+p, ... <= L_local
                    idxs = np.arange(start_chk, L_local + 1, p)
                    if len(idxs) > 0:
                        if np.min(counts[idxs]) > 1:
                            safe = True
                        else:
                            safe = False
                    else:
                        safe = True

                if safe:
                    apply_change(i, r_old, r_new)
                    residues[i] = r_new
                    
                    # Extend L
                    while L_local < MAX_L and counts[L_local + 1] > 0:
                        L_local += 1
                    extended = True
                    break # Restart search for next gap

        return L_local

    L = calc_L()
    best_L = L
    best_residues = residues[:]

    # Main Search Loop: Iterated Local Search with Simulated Annealing-like acceptance
    # We accept worse states with some probability to escape local minima.
    while time.time() < deadline:
        new_L = greedy_extend(L)
        
        if new_L > L:
            L = new_L
            if L > best_L:
                best_L = L
                best_residues = residues[:]
        else:
            # Stuck. Kick.
            save_residues = residues[:]
            save_counts = counts.copy()
            save_L = L
            
            # Kick: Change a few random residues
            K = random.randint(1, 3)
            for _ in range(K):
                i_kick = random.randint(0, N - 1)
                p_kick = selected_primes[i_kick]
                old_r_kick = residues[i_kick]
                new_r_kick = random.randint(0, p_kick - 1)
                
                if old_r_kick != new_r_kick:
                    apply_change(i_kick, old_r_kick, new_r_kick)
                    residues[i_kick] = new_r_kick
            
            # Repair
            L_temp = calc_L()
            L_repaired = greedy_extend(L_temp)
            
            # Acceptance Criteria
            # If better, accept.
            # If worse, accept with probability 0.1 (to allow exploration)
            # If much worse, reject.
            if L_repaired >= L:
                L = L_repaired
                if L > best_L:
                    best_L = L
                    best_residues = residues[:]
            else:
                # Check if it's a "small" drop, maybe worth keeping to escape
                # If drop is small, accept with prob 0.2
                if L - L_repaired < 50 and random.random() < 0.2:
                    L = L_repaired
                else:
                    # Revert
                    residues[:] = save_residues
                    counts[:] = save_counts
                    L = save_L

    return [[selected_primes[i], best_residues[i]] for i in range(N)]
