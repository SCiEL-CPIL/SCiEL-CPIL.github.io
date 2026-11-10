import time
import random
import sys

N = 5000
MAX_TIME = 280.0  # Safety margin before 330s kill

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

# Precompute residue masks
# residue_masks[p] is a list of p integers, where residue_masks[p][r] is a bitset
# with bit i set if i % p == r.
residue_masks = {}
for p in PRIMES:
    masks = [0] * p
    for x in range(N + 1):
        r = x % p
        masks[r] |= (1 << x)
    residue_masks[p] = masks

def _is_admissible_mask(mask, size):
    """Check if the set represented by mask is admissible."""
    if size == 0:
        return True
    for p in PRIMES:
        if p > size:
            break
        masks = residue_masks[p]
        # Check if all residues are covered
        covered_all = True
        for r in range(p):
            if (mask & masks[r]) == 0:
                covered_all = False
                break
        if covered_all:
            return False
    return True

def _greedy_bitset(rng, strategy=0, initial_density=1.0):
    """
    Greedy deletion using bitsets.
    strategy 0: Smallest violating prime, min count residue
    strategy 1: Random violating prime, min count residue
    strategy 2: Random violating prime, random among min count + 1
    strategy 3: Random violating prime, random residue
    """
    # Initialize mask
    if initial_density >= 1.0:
        mask = (1 << (N + 1)) - 1
    else:
        mask = 0
        for i in range(N + 1):
            if rng.random() < initial_density:
                mask |= (1 << i)
    
    size = mask.bit_count()
    
    while True:
        # Find violating primes
        violating_primes = []
        for p in PRIMES:
            if p > size:
                break
            masks = residue_masks[p]
            covered_all = True
            for r in range(p):
                if (mask & masks[r]) == 0:
                    covered_all = False
                    break
            if covered_all:
                violating_primes.append(p)
        
        if not violating_primes:
            break
        
        if strategy == 0:
            p = violating_primes[0]
        else:
            p = rng.choice(violating_primes)
        
        masks = residue_masks[p]
        counts = []
        for r in range(p):
            counts.append((mask & masks[r]).bit_count())
        
        min_count = min(counts)
        
        if strategy == 0 or strategy == 1:
            # Min count, tie-break lowest or random
            candidates = [r for r in range(p) if counts[r] == min_count]
            if strategy == 0:
                r_to_remove = candidates[0]
            else:
                r_to_remove = rng.choice(candidates)
        elif strategy == 2:
            # Random among min count + 1
            candidates = [r for r in range(p) if counts[r] <= min_count + 1]
            r_to_remove = rng.choice(candidates)
        else:
            # Random residue
            r_to_remove = rng.randint(0, p - 1)
            
        # Remove
        mask &= ~masks[r_to_remove]
        size -= counts[r_to_remove]
        
        if size == 0:
            break
            
    return mask, size

def _add_back_search(mask, size, rng, max_rounds=10, start_time=0, deadline=280):
    """
    Try to add back removed elements one at a time if it doesn't violate admissibility.
    """
    full_mask = (1 << (N + 1)) - 1
    removed_mask = full_mask ^ mask
    
    for round_num in range(max_rounds):
        if time.time() - start_time > deadline - 5:
            break
        
        # Get list of removed elements for shuffling
        # Extracting bits from a large int to a list is O(N)
        removed_list = []
        m = removed_mask
        i = 0
        while m:
            if m & 1:
                removed_list.append(i)
            m >>= 1
            i += 1
        
        if not removed_list:
            break
            
        rng.shuffle(removed_list)
        
        improved = False
        for x in removed_list:
            if time.time() - start_time > deadline - 5:
                break
            
            # Check if adding x is safe
            # new_size = size + 1
            # We need to check primes p <= size + 1
            # If p > size, then p must be size+1 (since p is prime and size+1 is the new size)
            # Actually, if p > size, the constraint for p is trivially satisfied unless size+1 == p and we cover all residues.
            # But if p > size, we have at most size residues, which is < p. So it's safe for p > size.
            # So we only need to check primes p <= size.
            # Wait, if size = 10, and we add 1, new_size = 11.
            # Primes to check: 2, 3, 5, 7, 11.
            # For p=11, we have 10 elements in mask. They cover at most 10 residues.
            # Adding x, we have 11 elements. They can cover at most 11 residues.
            # If they cover all 11, it's invalid.
            # So we MUST check p = size + 1 if it is prime.
            
            ok = True
            for p in PRIMES:
                if p > size + 1:
                    break
                
                # Optimization: If p > size + 1, break.
                # If p > size, then p == size + 1 (if prime).
                # If p <= size, we check as before.
                
                masks = residue_masks[p]
                
                # Check current coverage
                # We need to know if mask covers p-1 residues and x is the missing one.
                # Or if mask covers p residues (impossible if valid).
                
                covered_count = 0
                missing_residues = []
                for r in range(p):
                    if (mask & masks[r]) != 0:
                        covered_count += 1
                    else:
                        missing_residues.append(r)
                        
                if covered_count == p - 1:
                    # Only one residue missing
                    if len(missing_residues) == 1:
                        missing_r = missing_residues[0]
                        if x % p == missing_r:
                            ok = False
                            break
                # If covered_count < p - 1, adding x cannot make it p.
                # If covered_count == p, mask was already invalid.
            
            if ok:
                mask |= (1 << x)
                size += 1
                removed_mask ^= (1 << x)
                improved = True
                
        if not improved:
            break
            
    return mask, size

def construct():
    start_time = time.time()
    best_mask = 0
    best_size = 0
    
    rng = random.Random(42)
    
    # Strategy 0: Deterministic Greedy from full set
    t0 = time.time()
    mask, size = _greedy_bitset(rng, strategy=0, initial_density=1.0)
    if _is_admissible_mask(mask, size):
        best_mask = mask
        best_size = size
    
    # Add-back on best
    if time.time() - start_time < MAX_TIME - 10:
        mask, size = _add_back_search(best_mask, best_size, rng, max_rounds=5, start_time=start_time, deadline=MAX_TIME)
        if _is_admissible_mask(mask, size):
            if size > best_size:
                best_mask = mask
                best_size = size

    # Multi-restart search
    # We will try different strategies and initial densities
    num_restarts = 0
    while time.time() - start_time < MAX_TIME - 10:
        strategy = rng.randint(0, 3)
        # Occasionally start from a random subset to escape local optima
        if rng.random() < 0.2:
            density = rng.uniform(0.5, 0.9)
        else:
            density = 1.0
            
        mask, size = _greedy_bitset(rng, strategy=strategy, initial_density=density)
        
        # Verify admissibility (safety check)
        if not _is_admissible_mask(mask, size):
            # If invalid, try to repair by greedy removal?
            # Or just skip. Greedy should produce valid sets.
            # Let's skip invalid ones to be safe.
            continue
        
        # Add-back search
        if time.time() - start_time < MAX_TIME - 20:
            mask, size = _add_back_search(mask, size, rng, max_rounds=10, start_time=start_time, deadline=MAX_TIME)
            if _is_admissible_mask(mask, size):
                if size > best_size:
                    best_size = size
                    best_mask = mask
        
        num_restarts += 1
        
        # Periodic polish on the global best
        if num_restarts % 5 == 0 and time.time() - start_time < MAX_TIME - 30:
            polished_mask, polished_size = _add_back_search(best_mask, best_size, rng, max_rounds=20, start_time=start_time, deadline=MAX_TIME)
            if _is_admissible_mask(polished_mask, polished_size):
                if polished_size > best_size:
                    best_size = polished_size
                    best_mask = polished_mask

    # Convert best_mask to list
    result = []
    m = best_mask
    i = 0
    while m:
        if m & 1:
            result.append(i)
        m >>= 1
        i += 1
        
    # Final safety check
    if not _is_admissible_mask(best_mask, best_size):
        # Emergency: return empty or minimal valid set
        # A single element is always admissible
        return [0]
        
    return result

if __name__ == "__main__":
    # Run construct
    res = construct()
    print(len(res))
