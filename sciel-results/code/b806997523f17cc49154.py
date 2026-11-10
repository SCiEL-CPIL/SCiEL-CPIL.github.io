import time
import math
import random
import numpy as np
from fractions import Fraction

_CAP = 100000
_DEADLINE = 110  # Safety margin

def _sieve_primes(n):
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(counts):
    """Find largest L such that 1..L are covered (counts[i] > 0)."""
    # counts is a numpy array, index 0.._CAP
    # We need 1..L to have counts > 0
    # Find first index i >= 1 where counts[i] == 0
    # Using numpy: 
    # mask = counts[1:] == 0
    # if any(mask): return index of first True + 1
    # else: return _CAP
    
    # Efficient way:
    # np.where(counts[1:] == 0)[0]
    # But this creates an array. 
    # Let's try a loop if _CAP is small? No, 10^5 is large.
    # NumPy is better.
    zeros = np.where(counts[1:] == 0)[0]
    if len(zeros) > 0:
        return zeros[0]
    return _CAP

def construct():
    start_time = time.time()
    
    # 1. Generate primes
    primes = _sieve_primes(1000)
    
    # 2. Select subset of primes based on budget
    # Greedy: Smallest primes first (highest density)
    selected_primes = []
    budget = Fraction(1, 1)
    for p in primes:
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected_primes.append(p)
        else:
            # Try to see if we can swap? No, keep it simple first.
            # If we can't add this, we might be able to add a larger one later?
            # No, larger primes have smaller cost. 
            # Wait, 1/p decreases as p increases.
            # So if we can't add p, we might be able to add p_next.
            # But we want to maximize coverage. Smaller primes cover more.
            # So we should stick with the smallest primes that fit.
            # However, sometimes skipping a medium prime allows adding two larger ones?
            # Unlikely to help coverage density.
            pass
            
    # If we have very few primes, we might want to try different sets?
    # For now, use the greedy set.
    # Example: {2, 3, 7, 43}
    
    if not selected_primes:
        return []

    # 3. Local Search on Residues
    # Initialize residues
    residues = [p // 2 for p in selected_primes] # Random-ish init
    
    # Counts array: 0 means not covered.
    # Size _CAP + 1
    counts = np.zeros(_CAP + 1, dtype=np.int32)
    
    # Helper to update counts
    def update_counts(idx, old_r, new_r, p, delta):
        # delta is +1 or -1
        # Update multiples of p
        # Start index: old_r % p, step p
        # Note: indices in counts are 1.._CAP
        # Residue r means i % p == r.
        # If r == 0, indices are p, 2p, ...
        # If r > 0, indices are r, r+p, ...
        
        # Generate indices
        # np.arange(start, stop, step)
        
        # For old_r:
        start_old = old_r if old_r > 0 else p
        # Ensure start is within 1.._CAP
        if start_old <= _CAP:
            idxs_old = np.arange(start_old, _CAP + 1, p)
            counts[idxs_old] += delta * (-1) # Removing old coverage
            
        # For new_r:
        start_new = new_r if new_r > 0 else p
        if start_new <= _CAP:
            idxs_new = np.arange(start_new, _CAP + 1, p)
            counts[idxs_new] += delta
            
    # Initialize counts
    for i, p in enumerate(selected_primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= _CAP:
            idxs = np.arange(start, _CAP + 1, p)
            counts[idxs] += 1
            
    current_L = _compute_L(counts)
    best_L = current_L
    best_residues = residues.copy()
    
    # Local Search
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < _DEADLINE:
        improved = False
        # Try to improve by changing one residue
        # Pick a random prime index
        idx = random.randint(0, len(selected_primes) - 1)
        p = selected_primes[idx]
        old_r = residues[idx]
        
        # Try a new residue
        # Heuristic: Try residues that might align with current gaps?
        # Or just random.
        # Or try +/- 1.
        # Let's try a few candidates: random, +1, -1, 0, p//2
        candidates = [random.randint(0, p - 1), (old_r + 1) % p, (old_r - 1) % p]
        # Add 0 and p//2 sometimes?
        if random.random() < 0.2:
            candidates.append(0)
        if random.random() < 0.2:
            candidates.append(p // 2)
            
        for new_r in candidates:
            if new_r == old_r:
                continue
                
            # Apply change
            update_counts(idx, old_r, new_r, p, 1)
            
            new_L = _compute_L(counts)
            
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                best_residues[idx] = new_r
                improved = True
                no_improve_count = 0
                # Keep the change
                residues[idx] = new_r
                break
            else:
                # Revert change
                update_counts(idx, old_r, new_r, p, -1)
                
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart with random residues
                residues = [random.randint(0, p - 1) for p in selected_primes]
                counts.fill(0)
                for i, p in enumerate(selected_primes):
                    r = residues[i]
                    start = r if r > 0 else p
                    if start <= _CAP:
                        idxs = np.arange(start, _CAP + 1, p)
                        counts[idxs] += 1
                current_L = _compute_L(counts)
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                no_improve_count = 0
                max_no_improve = int(max_no_improve * 1.5) # Increase patience
            
    # Return best found
    return [[p, r] for p, r in zip(selected_primes, best_residues)]

# Ensure the function signature matches
# The prompt says: construct() -> classes [[p, r], ...]
# The code above does this.

# One issue: The greedy prime selection might not be optimal.
# What if we skip 7 and use 5 and 11?
# 1/2 + 1/3 + 1/5 + 1/11 + 1/17 ...
# 0.5 + 0.33 + 0.2 = 1.03. Can't use 5 if we use 2,3.
# So {2,3} is max for sum < 1?
# 1/2 + 1/3 = 0.833. Rem 0.166.
# Next smallest prime is 5 (0.2) - too big.
# 7 (0.142) - fits. Rem 0.024.
# 43 (0.023) - fits. Rem 0.001.
# 47 (0.021) - fits? 0.024 - 0.021 = 0.003.
# So {2, 3, 7, 47} is possible.
# My greedy algorithm adds primes in ascending order.
# 2, 3, 5 (no), 7 (yes), 11 (no, 1/11=0.09 > 0.024), ...
# It will skip 5, 11, 13... until it finds one that fits.
# This is correct.

# However, is {2, 3, 7, 43} better than {2, 3, 5, ...}?
# We can't use 5 with 2,3.
# So the set is determined by the budget constraint.
# The local search will optimize the residues for this set.

# Potential Issue: What if a different set of primes is better?
# E.g. {3, 5, 7, 11, 13, ...} without 2?
# Sum 1/3+1/5+1/7+1/11+1/13+1/17+1/19...
# 0.33+0.2+0.14+0.09+0.07+0.05+0.05 = 0.93.
# This uses many primes.
# Coverage with many primes might be better?
# The greedy set {2,3,7,43} uses 4 primes.
# The set without 2 uses ~7 primes.
# More primes = more flexibility in residues?
# But smaller primes (2,3) cover more integers per residue class.
# It's a trade-off.
# To be safe, I should try a few different prime sets.

def _select_primes_set(primes, target_budget=Fraction(1,1)):
    # Greedy ascending
    set1 = []
    b = target_budget
    for p in primes:
        c = Fraction(1, p)
        if b >= c:
            b -= c
            set1.append(p)
    return set1

def _select_primes_set_no2(primes, target_budget=Fraction(1,1)):
    # Greedy ascending, skip 2
    set2 = []
    b = target_budget
    for p in primes:
        if p == 2: continue
        c = Fraction(1, p)
        if b >= c:
            b -= c
            set2.append(p)
    return set2

def construct():
    start_time = time.time()
    primes = _sieve_primes(1000)
    
    # Try a few candidate sets
    candidate_sets = []
    
    # Set 1: Greedy with 2
    s1 = _select_primes_set(primes)
    if s1: candidate_sets.append(s1)
    
    # Set 2: Greedy without 2
    s2 = _select_primes_set_no2(primes)
    if s2: candidate_sets.append(s2)
    
    # Set 3: Greedy without 2,3
    s3 = []
    b = Fraction(1,1)
    for p in primes:
        if p < 5: continue
        c = Fraction(1, p)
        if b >= c:
            b -= c
            s3.append(p)
    if s3: candidate_sets.append(s3)
    
    best_L_global = 0
    best_classes_global = []
    
    for p_set in candidate_sets:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        # Local Search for this set
        residues = [random.randint(0, p - 1) for p in p_set]
        counts = np.zeros(_CAP + 1, dtype=np.int32)
        
        # Init counts
        for i, p in enumerate(p_set):
            r = residues[i]
            start = r if r > 0 else p
            if start <= _CAP:
                idxs = np.arange(start, _CAP + 1, p)
                counts[idxs] += 1
                
        current_L = _compute_L(counts)
        best_L = current_L
        best_residues = residues.copy()
        
        no_improve = 0
        max_no_improve = 100
        
        while time.time() - start_time < _DEADLINE - 5:
            improved = False
            idx = random.randint(0, len(p_set) - 1)
            p = p_set[idx]
            old_r = residues[idx]
            
            # Try candidates
            candidates = [random.randint(0, p - 1), (old_r + 1) % p, (old_r - 1) % p]
            
            for new_r in candidates:
                if new_r == old_r: continue
                
                # Update
                # Remove old
                start_old = old_r if old_r > 0 else p
                if start_old <= _CAP:
                    idxs_old = np.arange(start_old, _CAP + 1, p)
                    counts[idxs_old] -= 1
                    
                # Add new
                start_new = new_r if new_r > 0 else p
                if start_new <= _CAP:
                    idxs_new = np.arange(start_new, _CAP + 1, p)
                    counts[idxs_new] += 1
                    
                new_L = _compute_L(counts)
                
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                    best_residues[idx] = new_r
                    residues[idx] = new_r
                    improved = True
                    no_improve = 0
                    break
                else:
                    # Revert
                    # Remove new
                    if start_new <= _CAP:
                        counts[idxs_new] -= 1
                    # Add old
                    if start_old <= _CAP:
                        counts[idxs_old] += 1
                        
            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    # Restart
                    residues = [random.randint(0, p - 1) for p in p_set]
                    counts.fill(0)
                    for i, p in enumerate(p_set):
                        r = residues[i]
                        start = r if r > 0 else p
                        if start <= _CAP:
                            idxs = np.arange(start, _CAP + 1, p)
                            counts[idxs] += 1
                    current_L = _compute_L(counts)
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                    no_improve = 0
                    max_no_improve = int(max_no_improve * 1.2)
                    
        if best_L > best_L_global:
            best_L_global = best_L
            best_classes_global = [[p, r] for p, r in zip(p_set, best_residues)]
            
    return best_classes_global
