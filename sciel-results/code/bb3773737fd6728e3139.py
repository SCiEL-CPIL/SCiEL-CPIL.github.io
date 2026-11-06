import time
import random
import math
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 118  # Safety margin

    # Generate primes up to 1000
    def _primes_upto(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    all_primes = _primes_upto(1000)
    selected_primes = [p for p in all_primes if p >= 13]
    
    # Verify budget
    cost = sum(Fraction(1, p) for p in selected_primes)
    if cost > Fraction(1, 1):
        # Fallback to smaller set if needed, but 13+ is safe
        pass

    N = len(selected_primes)
    MAX_L = 100000  # Cap at 10^5 as specified

    # Initialize residues with diagonal strategy: r_i = i mod p
    # This ensures good initial coverage
    residues = [i % p for i, p in enumerate(selected_primes)]
    
    # Initialize counts using NumPy
    counts = np.zeros(MAX_L + 1, dtype=np.int16)
    
    for i in range(N):
        p = selected_primes[i]
        r = residues[i]
        counts[r::p] += 1

    def get_L():
        # Find first index i >= 1 where counts[i] == 0
        idx = np.argmax(counts[1:] == 0)
        return idx

    L = get_L()
    best_L = L
    best_residues = residues[:]
    
    # Greedy prefix extension: try to extend coverage one step at a time
    # For each uncovered position, find a prime that can cover it with minimal disruption
    while time.time() < deadline:
        # Check if we've reached MAX_L
        if L >= MAX_L:
            break
        
        # Find the first uncovered position
        first_gap = L + 1
        
        # Try to cover first_gap by adjusting one prime's residue
        found = False
        for i in range(N):
            p = selected_primes[i]
            old_r = residues[i]
            
            # For prime p, we need r such that first_gap ≡ r (mod p)
            new_r = first_gap % p
            
            if new_r == old_r:
                # Already covers first_gap, shouldn't happen if L is correct
                continue
            
            # Check if changing to new_r would break coverage of 1..L
            # We need to verify that all positions 1..L remain covered
            # After change, position x is covered if:
            # - x was covered by prime i and x ≡ new_r (mod p), OR
            # - x was covered by some other prime j
            
            # Efficient check: only positions in old_r::p that are NOT in new_r::p
            # might become uncovered
            # Positions in new_r::p that were not in old_r::p might become newly covered
            
            # Simpler: just apply the change, check L, revert if worse
            counts[old_r::p] -= 1
            counts[new_r::p] += 1
            
            new_L = get_L()
            
            if new_L >= L:
                # Accept this change
                residues[i] = new_r
                L = new_L
                if L > best_L:
                    best_L = L
                    best_residues = residues[:]
                found = True
                break
            else:
                # Revert
                counts[new_r::p] -= 1
                counts[old_r::p] += 1
        
        if not found:
            # Can't extend with single change, try local search around current L
            # Use simulated annealing for a limited time
            T = 5.0
            T_min = 0.1
            cooling_rate = 0.9995
            stuck_count = 0
            
            while time.time() < deadline and stuck_count < 50:
                i = random.randint(0, N - 1)
                p = selected_primes[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                
                if old_r == new_r:
                    continue
                
                counts[old_r::p] -= 1
                counts[new_r::p] += 1
                
                new_L = get_L()
                
                if new_L > L:
                    L = new_L
                    residues[i] = new_r
                    if L > best_L:
                        best_L = L
                        best_residues = residues[:]
                    stuck_count = 0
                else:
                    delta = L - new_L
                    if delta == 0:
                        if random.random() < 0.3:
                            L = new_L
                            residues[i] = new_r
                        else:
                            counts[new_r::p] -= 1
                            counts[old_r::p] += 1
                    else:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            L = new_L
                            residues[i] = new_r
                        else:
                            counts[new_r::p] -= 1
                            counts[old_r::p] += 1
                
                T *= cooling_rate
                if T < T_min:
                    T = T_min
                    stuck_count += 1
                elif new_L < L:
                    stuck_count += 1
                else:
                    stuck_count = 0
        
        # If L didn't change after greedy attempt, we might be stuck
        # Try more aggressive local search
        if L == best_L:
            # Continue with simulated annealing until deadline
            T = 10.0
            T_min = 0.01
            cooling_rate = 0.999
            
            while time.time() < deadline:
                i = random.randint(0, N - 1)
                p = selected_primes[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                
                if old_r == new_r:
                    continue
                
                counts[old_r::p] -= 1
                counts[new_r::p] += 1
                
                new_L = get_L()
                
                if new_L > L:
                    L = new_L
                    residues[i] = new_r
                    if L > best_L:
                        best_L = L
                        best_residues = residues[:]
                else:
                    delta = L - new_L
                    if delta == 0:
                        if random.random() < 0.5:
                            L = new_L
                            residues[i] = new_r
                        else:
                            counts[new_r::p] -= 1
                            counts[old_r::p] += 1
                    else:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            L = new_L
                            residues[i] = new_r
                        else:
                            counts[new_r::p] -= 1
                            counts[old_r::p] += 1
                
                T *= cooling_rate
                if T < T_min:
                    T = T_min

    return [[selected_primes[i], best_residues[i]] for i in range(N)]
