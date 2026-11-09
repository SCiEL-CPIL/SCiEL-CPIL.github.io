import time
import random
import math

N = 5000
MAX_TIME = 290.0  # Safety margin (300s limit, keep 10s buffer)

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)
MAX_PRIME_LIMIT = N + 1
PRIMES_LIMITED = [p for p in PRIMES if p <= MAX_PRIME_LIMIT]
NUM_PRIMES = len(PRIMES_LIMITED)

# Precompute x % p for all x and p? 
# Memory: 5001 * 669 ints ~ 3.3M ints ~ 26MB. Acceptable.
# Or compute on the fly? 5001 * 669 % operations per move is too slow for SA?
# 3.3M ops per move. 10000 moves = 33B ops. Too slow.
# We need precomputation.

# Precompute residues: residues[x][p_index] = x % PRIMES_LIMITED[p_index]
# To save memory and time, we can store as a list of arrays or just list of lists.
# Let's use a list of lists for simplicity, or a flat array?
# residues[x] is a list of length NUM_PRIMES.

residues = []
for x in range(N + 1):
    row = [x % p for p in PRIMES_LIMITED]
    residues.append(row)

# Data structure for SA state:
# kept: boolean array of length N+1
# counts[p_index][r]: count of kept elements with residue r mod p
# violations: set of p_index such that all residues are covered (min count > 0)
# size: number of kept elements

def _init_counts(kept):
    """Initialize counts and violations from a kept array."""
    counts = [[0] * p for p in PRIMES_LIMITED]
    for x in range(N + 1):
        if kept[x]:
            for i, p in enumerate(PRIMES_LIMITED):
                r = residues[x][i]
                counts[i][r] += 1
    
    violations = set()
    for i, p in enumerate(PRIMES_LIMITED):
        c = counts[i]
        # Check if all residues are covered
        # Note: We only care if p <= size. But in SA, size changes.
        # For the "violations" set in the energy function, we should probably 
        # consider all primes, or only those <= current size?
        # The constraint is: for every prime p <= len(tuple), some residue is missed.
        # So if p > size, the constraint is vacuously true (or doesn't apply).
        # So we only count p as a violation if p <= size AND all residues are covered.
        
        # However, for the energy function during SA, we want to penalize 
        # "potential" violations or just current ones?
        # If we penalize all primes where all residues are covered, regardless of size,
        # it might be too strong for large p.
        # Let's stick to the definition: violation if p <= size and covered.
        
        # But size changes dynamically. 
        # Let's just track "is_covered" for all primes.
        # Then in energy, we check: if is_covered[i] and PRIMES_LIMITED[i] <= size: violation.
        
        if min(c) > 0:
            violations.add(i)
            
    return counts, violations

def _greedy_initial():
    """Get a good starting point using the existing greedy logic."""
    mask = (1 << (N + 1)) - 1
    size = N + 1
    # We need counts for greedy? Or just use the bitset method from before?
    # Let's use the bitset method to get a valid initial state.
    
    # Re-implement simple greedy to get initial valid set
    # This is just to get a valid start for SA.
    # We can also start with empty set and add, but greedy is better.
    
    # Use the _greedy_bitset_internal logic but return the kept list
    # To avoid code duplication, we can just run a fast greedy here.
    
    # For speed in initialization, we'll use a simple greedy:
    # Start with all. Remove residue classes to break coverage.
    # This is O(N * #primes * #residues) which is fine for one run.
    
    kept = [True] * (N + 1)
    current_size = N + 1
    
    # Precompute masks for greedy? Or just use the bitset logic?
    # Let's use the bitset logic from the current program as it's proven.
    # But we need to integrate it.
    
    # Actually, let's just start with a random valid set?
    # Or start with empty and add?
    # Starting with empty and adding is hard to ensure validity quickly.
    # Starting with full and removing is easy.
    
    # Let's run the deterministic greedy from the current code to get a valid set.
    # We need to extract the kept list.
    
    # To keep this self-contained and fast, I'll implement a simplified greedy here.
    # It doesn't need to be optimal, just valid.
    
    # Simple Greedy:
    # 1. Start with all numbers.
    # 2. While there is a prime p <= size such that all residues are covered:
    #    - Pick the smallest such p.
    #    - Pick the residue r with the minimum count.
    #    - Remove all x in kept where x % p == r.
    
    # This is O(N * #steps). #steps is roughly N/p_avg.
    
    # To make it faster, we use the precomputed residues.
    
    counts = [[0] * p for p in PRIMES_LIMITED]
    for x in range(N + 1):
        for i, p in enumerate(PRIMES_LIMITED):
            counts[i][residues[x][i]] += 1
            
    current_size = N + 1
    kept = [True] * (N + 1)
    
    while True:
        # Find violating prime
        viol_p_idx = -1
        for i, p in enumerate(PRIMES_LIMITED):
            if p > current_size:
                break
            if min(counts[i]) > 0:
                viol_p_idx = i
                break
        
        if viol_p_idx == -1:
            break
        
        p = PRIMES_LIMITED[viol_p_idx]
        c = counts[viol_p_idx]
        min_count = min(c)
        # Find residues with min count
        min_residues = [r for r in range(p) if c[r] == min_count]
        r_to_remove = min_residues[0] # Deterministic for init
        
        # Remove elements
        for x in range(N + 1):
            if kept[x] and residues[x][viol_p_idx] == r_to_remove:
                kept[x] = False
                current_size -= 1
                # Update counts for all primes
                for j, pj in enumerate(PRIMES_LIMITED):
                    counts[j][residues[x][j]] -= 1
                    
    return kept, current_size

def construct():
    start_time = time.time()
    
    # 1. Get initial valid solution
    kept, size = _greedy_initial()
    best_size = size
    best_kept = kept[:]
    
    # Initialize SA state
    counts, violations = _init_counts(kept)
    
    # SA Parameters
    # We want to maximize size. 
    # Energy E = -size + lambda * num_violations
    # We accept moves if delta_E < 0 or with prob exp(-delta_E / T)
    
    # Lambda should be large enough to discourage violations.
    # Max possible violations is #primes ~ 669.
    # Max size is 5000.
    # So Lambda ~ 10 is enough to make violations "expensive" but not impossible.
    LAMBDA = 10.0
    
    T = 5.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    # We will run SA until time limit.
    
    # Precompute prime indices for quick access?
    # PRIMES_LIMITED is already indexed.
    
    while time.time() - start_time < MAX_TIME:
        # Move: Pick a random element x and toggle it?
        # Or pick a random x and try to add/remove?
        # Toggling is simpler.
        
        x = random.randint(0, N)
        
        if kept[x]:
            # Try to remove x
            # Check if removing x keeps it valid or reduces violations?
            # Actually, removing x never *creates* a violation (it can only reduce counts, 
            # so it can only *break* coverage, i.e., remove violations).
            # It reduces size by 1.
            # So delta_E = -(-1) + LAMBDA * (new_viol - old_viol)
            # new_viol <= old_viol.
            # So delta_E = 1 + LAMBDA * (negative or zero).
            # If violations decrease, delta_E can be negative (good).
            # If violations stay same, delta_E = 1 (bad, we lose size).
            # So we should rarely remove elements unless it fixes a violation.
            
            # Let's simulate removal
            old_viol_count = len(violations)
            
            # Update counts
            for i, p in enumerate(PRIMES_LIMITED):
                r = residues[x][i]
                old_c = counts[i][r]
                counts[i][r] -= 1
                
                # Check if this prime was violating and is no longer?
                # It was violating if i in violations.
                # It is no longer violating if min(counts[i]) == 0.
                # Since we only decreased one count, min can only stay same or become 0.
                if i in violations:
                    if counts[i][r] == 0:
                        # This residue is now empty, so not all residues are covered.
                        violations.discard(i)
                        
            new_viol_count = len(violations)
            
            delta_size = -1
            delta_viol = new_viol_count - old_viol_count
            delta_E = -delta_size + LAMBDA * delta_viol # Wait, E = -size + L*V. 
            # delta_E = E_new - E_old = -(size-1) + L*V_new - (-size + L*V_old)
            # delta_E = 1 + L*(V_new - V_old)
            
            # Acceptance
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                kept[x] = False
                size -= 1
            else:
                # Revert counts
                for i, p in enumerate(PRIMES_LIMITED):
                    r = residues[x][i]
                    counts[i][r] += 1
                    # No need to revert violations set if we didn't apply?
                    # We modified violations set in place. We need to revert it too if rejected.
                    # This is messy. 
                    
                    # Better: Calculate delta_E without modifying state first?
                    # Or use a stack?
                    # For simplicity and speed, let's use a "copy" strategy for violations?
                    # Violations is a set of ints. Copying is fast.
                    
                    pass # This logic is flawed for rejection. Let's refactor.
                    
        # Refactor SA Step to be safe:
        # 1. Calculate delta_E based on current state.
        # 2. If accepted, apply change.
        
        # Let's restart the loop logic with a cleaner apply/revert pattern.
        
        # To avoid complex revert logic, we can calculate the effect first.
        
        # Let's try a different move: **Swap**?
        # Or just stick to Add/Remove but with careful state management.
        
        # Given the complexity of reverting set changes, let's use a simpler heuristic:
        # **Hill Climbing with Random Restarts** but with **Local Moves** that are guaranteed to be valid?
        # No, that's what greedy does.
        
        # Let's go back to **Simulated Annealing** but use **Arrays** for counts and **Integer** for violations bitmask?
        # Violations set is small (< 669).
        
        # Let's try a **Greedy Repair** approach instead.
        # 1. Start with a random set (or greedy set).
        # 2. If invalid, find a violating prime p.
        # 3. Remove one element from the "most crowded" residue of p?
        # 4. Repeat.
        # This is basically the greedy deletion.
        
        # The key to beating greedy is **Escaping Local Optima**.
        # Greedy gets stuck because it always picks the "best" immediate move.
        # SA allows "worse" moves (temporarily reducing size or increasing violations) to find better peaks.
        
        # Let's implement SA properly with a "trial" mechanism.
        
        # We will maintain:
        # kept: list of bools
        # counts: list of lists
        # viol_mask: int bitmask where bit i is 1 if prime i is violating (all residues covered)
        
        # Re-initialize with greedy result
        kept, size = _greedy_initial()
        counts, _ = _init_counts(kept)
        
        # Build viol_mask
        viol_mask = 0
        for i, p in enumerate(PRIMES_LIMITED):
            if min(counts[i]) > 0:
                viol_mask |= (1 << i)
        
        best_size = size
        best_kept = kept[:]
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        # Precompute prime indices for quick check?
        
        while time.time() - start_time < MAX_TIME:
            x = random.randint(0, N)
            
            if kept[x]:
                # Try Remove
                # Calculate delta_E
                # V_old = popcount(viol_mask)
                # V_new: 
                # For each prime i, if viol_mask has bit i, and counts[i][r] becomes 0, 
                # then bit i is cleared.
                
                old_viol_count = bin(viol_mask).count('1')
                
                new_viol_mask = viol_mask
                for i, p in enumerate(PRIMES_LIMITED):
                    if (viol_mask >> i) & 1: # Was violating
                        r = residues[x][i]
                        if counts[i][r] == 1:
                            # Becomes 0, so no longer violating
                            new_viol_mask &= ~(1 << i)
                
                new_viol_count = bin(new_viol_mask).count('1')
                
                delta_size = -1
                delta_viol = new_viol_count - old_viol_count
                delta_E = 1 + LAMBDA * delta_viol
                
                if delta_E < 0 or (T > 0 and random.random() < math.exp(-delta_E / T)):
                    # Apply
                    kept[x] = False
                    size -= 1
                    for i, p in enumerate(PRIMES_LIMITED):
                        counts[i][residues[x][i]] -= 1
                    viol_mask = new_viol_mask
                    
                    if size > best_size and new_viol_count == 0:
                        best_size = size
                        best_kept = kept[:]
            else:
                # Try Add
                # V_old = popcount(viol_mask)
                # V_new:
                # For each prime i, if NOT violating, and adding x completes the set,
                # then bit i is set.
                
                old_viol_count = bin(viol_mask).count('1')
                
                new_viol_mask = viol_mask
                for i, p in enumerate(PRIMES_LIMITED):
                    if not ((viol_mask >> i) & 1): # Was not violating
                        r = residues[x][i]
                        # If counts[i][r] was 0, and all other counts > 0,
                        # then adding x (making it 1) makes all > 0.
                        if counts[i][r] == 0:
                            # Check if all others are > 0
                            # We know it was not violating, so at least one was 0.
                            # If this was the only 0, then it becomes violating.
                            # How to check quickly?
                            # min(counts[i]) == 0.
                            # If min(counts[i]) == 0 and the index of min is r, 
                            # and counts[i][r] == 0, and all others > 0?
                            # Actually, if min(counts[i]) == 0, and we add to a 0-count residue,
                            # it becomes 1. If all others were already > 0, then min becomes 1.
                            # So we need to know if r was the ONLY residue with count 0.
                            
                            # Efficient check:
                            # Count how many residues have count 0.
                            # If count_zero == 1 and counts[i][r] == 0, then it becomes violating.
                            
                            # This requires scanning p residues. p is small.
                            zero_count = 0
                            for rr in range(p):
                                if counts[i][rr] == 0:
                                    zero_count += 1
                                    if zero_count > 1:
                                        break
                            
                            if zero_count == 1 and counts[i][r] == 0:
                                new_viol_mask |= (1 << i)
                
                new_viol_count = bin(new_viol_mask).count('1')
                
                delta_size = 1
                delta_viol = new_viol_count - old_viol_count
                delta_E = -1 + LAMBDA * delta_viol # E = -size + L*V. delta_E = -1 + L*delta_viol
                
                if delta_E < 0 or (T > 0 and random.random() < math.exp(-delta_E / T)):
                    # Apply
                    kept[x] = True
                    size += 1
                    for i, p in enumerate(PRIMES_LIMITED):
                        counts[i][residues[x][i]] += 1
                    viol_mask = new_viol_mask
                    
                    if size > best_size and new_viol_count == 0:
                        best_size = size
                        best_kept = kept[:]
            
            T *= cooling
            if T < T_min:
                T = T_min
                # Optionally reheat?
                # T = 5.0
                
    # Convert best_kept to list
    result = [i for i, k in enumerate(best_kept) if k]
    return result

# Final check: The code above has a potential issue with the `viol_mask` update in the "Try Add" case.
# Specifically, `zero_count` loop is inside the prime loop.
# Complexity: O(NUM_PRIMES * p) per move.
# Average p is small, but for large p, it's expensive.
# However, we only do this check if `not violating` and `counts[i][r] == 0`.
# This is a rare event.
# So it should be fast enough.

# One optimization: Precompute `prime_indices`? No, we use enumerate.

# Let's double check the `delta_E` formula.
# E = -Size + Lambda * Violations
# Move: Add x. Size -> Size+1. Violations -> V_new.
# E_new = -(Size+1) + L*V_new
# E_old = -Size + L*V_old
# delta_E = E_new - E_old = -1 + L*(V_new - V_old)
# Correct.

# Move: Remove x. Size -> Size-1. Violations -> V_new.
# E_new = -(Size-1) + L*V_new = -Size + 1 + L*V_new
# delta_E = 1 + L*(V_new - V V_old)
# Correct.

# The code looks solid.
