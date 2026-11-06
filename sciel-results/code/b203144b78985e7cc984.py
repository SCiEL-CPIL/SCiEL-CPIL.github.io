"""
Strategy: Simulated Annealing over joint prime selection and residues.
1. Identify prime sets that maximize coverage potential (prioritizing small primes).
2. Use NumPy for fast coverage evaluation (vectorized modulo checks).
3. Optimize residues and prime set membership via SA to escape local minima.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _get_primes(limit=1000):
    return [p for p in range(2, limit + 1) if _is_prime(p)]

# Precompute primes
ALL_PRIMES = _get_primes(1000)
P_ARRAY = np.array(ALL_PRIMES)
N_PRIMES = len(ALL_PRIMES)

# Precompute coverage masks for each prime? 
# No, 168 primes * 100000 booleans is 16.8MB, which is fine.
# But we need to sum them. 
# Better: for a given set of active primes and residues, compute coverage.
# Vectorized approach:
# For each active prime i, create a boolean mask M_i where M_i[n] = (n % p_i == r_i).
# Covered = OR of M_i.
# We can do this in a loop over active primes, which is fast if active primes are few.

def _compute_coverage_fast(active_indices, residues, cap=100000):
    """
    Returns the largest L such that 1..L are covered.
    active_indices: list of indices into ALL_PRIMES
    residues: list of residues corresponding to active_indices
    """
    if not active_indices:
        return 0
    
    # We need to check 1..cap.
    # Instead of allocating a full boolean array of size cap+1 every time (expensive),
    # we can use a trick or just accept the allocation cost if it's fast enough.
    # 100000 booleans is ~100KB. Allocation is fast.
    covered = np.zeros(cap + 1, dtype=bool)
    
    # Range 1..cap
    # n = 1..cap
    # We can use np.arange(1, cap+1)
    # But creating arange every time is also costly.
    # Let's precompute the range?
    # Actually, let's just use the property:
    # n % p == r  <=>  n is in {r, r+p, r+2p, ...}
    # We can mark these in the covered array.
    
    for idx, r in zip(active_indices, residues):
        p = ALL_PRIMES[idx]
        # First number >= 1 with n % p == r
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p * ((1 - start + p - 1) // p)
        
        # Mark start, start+p, ... <= cap
        # Use numpy slice? No, step is p.
        # covered[start::p] = True
        # This is very fast in numpy.
        covered[start::p] = True
        
    # Find longest prefix of True starting from 1
    # covered[1] must be True, covered[2] must be True, etc.
    # We can find the first False in covered[1:]
    # If all True, L = cap.
    # Use np.argmax on ~covered[1:] to find first False index.
    # If no False, argmax returns 0 (if all True, ~covered is all False, argmax is 0? No, argmax of all False is 0).
    # Wait, if all are True, ~covered is all False. np.argmax([False, False]) returns 0.
    # So L = 1 + 0 = 1? No.
    # If covered[1:] is all True, we want L = cap.
    # If there is a False at index k in covered[1:], then L = 1 + k.
    # Example: covered = [F, T, T, F, ...] (indices 0,1,2,3)
    # covered[1:] = [T, T, F]
    # ~covered[1:] = [F, F, T]
    # argmax = 2. L = 1 + 2 = 3. Correct.
    
    if covered[1].any(): # Check if 1 is covered? 
        # Actually, if 1 is not covered, L=0.
        # covered[0] is always False (we start from 1).
        # If covered[1] is False, L=0.
        if not covered[1]:
            return 0
        
    # Find first False in covered[1:]
    # If all True, return cap.
    # np.argmax on a boolean array returns the index of the first True.
    # We want the first False in covered, so first True in ~covered.
    # But we only care about 1..cap.
    # Let's invert covered[1:]
    inv = ~covered[1:]
    # If any True in inv, find index.
    # If no True, L = cap.
    idx_false = np.argmax(inv)
    if inv[idx_false]:
        L = 1 + idx_false
    else:
        L = cap
        
    return L

def _validate_budget(active_indices):
    """Check if sum(1/p) <= 1 using exact fractions."""
    total = Fraction(0, 1)
    for idx in active_indices:
        total += Fraction(1, ALL_PRIMES[idx])
    return total <= 1

def construct():
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: return a known valid solution
        # Seed solution: primes 13..997
        primes = [p for p in range(13, 1001) if _is_prime(p)]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]

    rng = random.Random(42)
    
    # Initialize with a good set of primes
    # Small primes are very effective.
    # Let's start with a set that includes 2, 3, 5, 7 if possible.
    # 1/2+1/3+1/5+1/7 = 1.175 > 1.
    # Try {2, 3, 5}: 1.033 > 1.
    # Try {2, 3, 7}: 0.975 < 1.
    # Try {2, 5, 7}: 0.842 < 1.
    # Try {3, 5, 7}: 0.675 < 1.
    
    # Let's start with {2, 3, 7} and add more primes if budget allows.
    # Or just start with a random valid set and let SA find the good one.
    # Better: Start with the set that maximizes "potential" (small primes).
    
    # Let's manually construct a strong initial state.
    # Primes: 2, 3, 7, 11, 13, ...
    # We want to maximize coverage.
    
    # Let's use a greedy initialization for prime selection:
    # Sort primes by 1/p (descending).
    # Add prime if budget allows.
    # This might pick 2, 3, 7, 11, 13...
    
    # 1/2 = 0.5
    # 1/3 = 0.333 (Total 0.833)
    # 1/7 = 0.142 (Total 0.975)
    # 1/11 = 0.09 (Total 1.065) -> Skip
    # 1/13 = 0.076 (Total 1.051) -> Skip
    # 1/17 = 0.058 (Total 1.033) -> Skip
    # 1/19 = 0.052 (Total 1.027) -> Skip
    # 1/23 = 0.043 (Total 1.018) -> Skip
    # 1/29 = 0.034 (Total 1.009) -> Skip
    # 1/31 = 0.032 (Total 1.007) -> Skip
    # 1/37 = 0.027 (Total 1.002) -> Skip
    # 1/41 = 0.024 (Total 0.999) -> OK!
    
    # So {2, 3, 7, 41} is a valid set.
    # Let's check coverage of {2, 3, 7, 41} with optimal residues.
    
    initial_primes = [2, 3, 7, 41]
    initial_indices = [ALL_PRIMES.index(p) for p in initial_primes]
    
    # Initialize residues randomly or heuristically
    # Heuristic: Try to cover small numbers.
    # Residues for 2, 3, 7, 41
    # Let's just start with random residues and let SA optimize.
    current_residues = [rng.randrange(p) for p in initial_primes]
    current_active = initial_indices[:]
    
    # Check budget
    if not _validate_budget(current_active):
        # Fallback to seed if initialization fails (shouldn't happen)
        primes = [p for p in range(13, 1001) if _is_prime(p)]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]
        
    best_L = _compute_coverage_fast(current_active, current_residues)
    best_active = current_active[:]
    best_residues = current_residues[:]
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    while time.time() - _START_TIME < _TIME_LIMIT and T > min_T:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Change residue of a random active prime
            if not current_active:
                continue
            pos = rng.randrange(len(current_active))
            idx = current_active[pos]
            p = ALL_PRIMES[idx]
            old_r = current_residues[pos]
            new_r = rng.randrange(p)
            
            if new_r == old_r:
                continue
                
            current_residues[pos] = new_r
            new_L = _compute_coverage_fast(current_active, current_residues)
            
            delta = new_L - best_L
            if delta > 0 or rng.random() < np.exp(delta / T):
                if delta > 0:
                    best_L = new_L
                    best_active = current_active[:]
                    best_residues = current_residues[:]
            else:
                current_residues[pos] = old_r
                
        elif move_type < 0.8:
            # Add a random prime (if budget allows)
            # Pick a random prime not in active set
            active_set = set(current_active)
            candidates = [i for i in range(N_PRIMES) if i not in active_set]
            if not candidates:
                continue
            new_idx = rng.choice(candidates)
            new_p = ALL_PRIMES[new_idx]
            
            # Check budget
            # We need to check if adding new_p keeps sum(1/p) <= 1
            # To speed up, we can maintain a running sum? 
            # Or just check with Fraction. It's slow but safe.
            # Optimization: Precompute reciprocals as Fractions?
            # Let's just check.
            test_active = current_active + [new_idx]
            if _validate_budget(test_active):
                current_active.append(new_idx)
                current_residues.append(rng.randrange(new_p))
                new_L = _compute_coverage_fast(current_active, current_residues)
                
                delta = new_L - best_L
                if delta > 0 or rng.random() < np.exp(delta / T):
                    if delta > 0:
                        best_L = new_L
                        best_active = current_active[:]
                        best_residues = current_residues[:]
                else:
                    current_active.pop()
                    current_residues.pop()
            else:
                # Try to swap? No, just skip.
                pass
                
        else:
            # Remove a random active prime (if > 1)
            if len(current_active) <= 1:
                continue
            pos = rng.randrange(len(current_active))
            removed_idx = current_active[pos]
            removed_r = current_residues[pos]
            
            current_active.pop(pos)
            current_residues.pop(pos)
            
            new_L = _compute_coverage_fast(current_active, current_residues)
            
            delta = new_L - best_L
            if delta > 0 or rng.random() < np.exp(delta / T):
                if delta > 0:
                    best_L = new_L
                    best_active = current_active[:]
                    best_residues = current_residues[:]
            else:
                # Revert
                # Insert back at pos
                current_active.insert(pos, removed_idx)
                current_residues.insert(pos, removed_r)
                
        T *= cooling_rate
        
    # Final check and return
    # Ensure best solution is valid
    if not _validate_budget(best_active):
        # Fallback
        primes = [p for p in range(13, 1001) if _is_prime(p)]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]
        
    result = [[ALL_PRIMES[i], r] for i, r in zip(best_active, best_residues)]
    return result
# EVOLVE-BLOCK-END
