import time
import math
import random
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 145  # Safety margin

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
    
    # Candidate prime sets with sum(1/p) <= 1
    # Set A: {3, 5} + primes >= 13. Sum ~ 0.97.
    # Set B: {2} + primes >= 67. Sum ~ 0.95.
    # Set C: {3, 5, 7} + primes >= 150. Sum ~ 0.99.
    candidate_sets = [
        [3, 5] + [p for p in all_primes if p >= 13],
        [2] + [p for p in all_primes if p >= 67],
        [3, 5, 7] + [p for p in all_primes if p >= 150]
    ]
    
    valid_sets = []
    for P in candidate_sets:
        s = sum(Fraction(1, p) for p in P)
        if s <= 1:
            valid_sets.append(P)
            
    if not valid_sets:
        # Fallback
        primes_seed = [p for p in all_primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]

    best_L_global = 0
    best_result_global = []
    
    # Allocate time among sets
    # 3 sets. Let's give more time to the first one as it has more small primes (denser coverage potential per prime).
    # But actually, more primes = more flexibility.
    # Set A has ~160 primes. Set B has ~100 primes. Set C has ~60 primes.
    # Let's split time equally or slightly favor A.
    time_per_set = 45  # 3 * 45 = 135s. 10s margin.
    
    L_MAX = 20000  # Use a larger window for better L estimation during search
    
    for P in valid_sets:
        if time.time() > deadline - 10:
            break
            
        set_start = time.time()
        set_deadline = set_start + time_per_set
        
        P_sorted = sorted(P)
        N = len(P_sorted)
        
        # 1. Greedy Initialization
        # Initialize residues to random values? Or 0?
        # Random is better for diversity.
        residues = [random.randint(0, p - 1) for p in P_sorted]
        
        # Initialize counts using NumPy
        counts = np.zeros(L_MAX + 1, dtype=np.int16)
        
        for i in range(N):
            p = P_sorted[i]
            r = residues[i]
            counts[r::p] += 1
            
        def get_L():
            # Find first index i >= 1 where counts[i] == 0
            # counts[1:] == 0 returns boolean array
            # argmax returns first True index. If no True, returns len.
            idx = np.argmax(counts[1:] == 0)
            return idx

        L = get_L()
        best_set_L = L
        best_set_residues = residues[:]
        
        # 2. Gap-Targeted Search with Random Jumps
        # Strategy:
        # - Find the first gap (first uncovered number).
        # - Try to cover it by changing the residue of one of the primes that *could* cover it (i.e., any prime).
        # - Actually, any prime can cover any number by choosing the right residue.
        # - So, pick a random prime, set its residue to cover the gap.
        # - If this improves L, accept. If it worsens, accept with prob exp(-delta/T).
        # - Also, mix in random jumps to escape local minima.
        
        T = 5.0
        T_min = 0.1
        cooling = 0.9999  # Slower cooling to allow more exploration
        
        # Precompute prime indices
        # We just pick random i
        
        while time.time() < set_deadline:
            # Determine move type
            # 50% Gap-Targeted, 50% Random Jump
            if random.random() < 0.5:
                # Gap-Targeted Move
                # Find first gap
                gap_pos = get_L() + 1  # First uncovered number
                if gap_pos > L_MAX:
                    # All covered up to L_MAX, do random jump
                    i = random.randint(0, N - 1)
                    p = P_sorted[i]
                    old_r = residues[i]
                    new_r = random.randint(0, p - 1)
                else:
                    # Pick a random prime to try to cover the gap
                    i = random.randint(0, N - 1)
                    p = P_sorted[i]
                    old_r = residues[i]
                    # Set residue so that p divides (gap_pos - r)
                    # gap_pos ≡ r (mod p) => r = gap_pos % p
                    new_r = gap_pos % p
                    
                    if new_r == old_r:
                        # No change, try another prime or random jump
                        continue
            else:
                # Random Jump
                i = random.randint(0, N - 1)
                p = P_sorted[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue
            
            # Apply change
            counts[old_r::p] -= 1
            counts[new_r::p] += 1
            
            new_L = get_L()
            
            # Acceptance
            if new_L > L:
                L = new_L
                residues[i] = new_r
                if L > best_set_L:
                    best_set_L = L
                    best_set_residues = residues[:]
            else:
                delta = L - new_L
                if delta == 0:
                    if random.random() < 0.5:
                        L = new_L
                        residues[i] = new_r
                    else:
                        counts[old_r::p] += 1
                        counts[new_r::p] -= 1
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        L = new_L
                        residues[i] = new_r
                    else:
                        counts[old_r::p] += 1
                        counts[new_r::p] -= 1
                        
            T *= cooling
            if T < T_min:
                T = T_min

        # Update global best
        if best_set_L > best_L_global:
            best_L_global = best_set_L
            best_result_global = [[P_sorted[i], best_set_residues[i]] for i in range(N)]
            
            # Check if we should break early
            if best_L_global >= 5000:
                break

    # Final check and return
    if not best_result_global:
        # Fallback to seed
        primes_seed = [p for p in all_primes if p >= 13]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    return best_result_global
