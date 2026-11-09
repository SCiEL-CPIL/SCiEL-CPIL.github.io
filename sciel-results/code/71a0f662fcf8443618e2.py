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

PRIMES = _primes(N)

# Precompute residue masks for each prime
# residue_masks[p_index][r] is a bitset (int) where bit i is 1 if i % p == r
# We only need primes up to N, but the constraint is p <= len(tuple).
# len(tuple) <= N+1 = 5001.
# So we need primes up to 5001.
MAX_PRIME_LIMIT = N + 1
PRIMES_LIMITED = [p for p in PRIMES if p <= MAX_PRIME_LIMIT]

# To save memory and time, we can store masks in a list of lists.
# p can be up to 5000. Number of primes ~ 669.
# Total masks ~ sum(p for p in primes) ~ 100,000 masks.
# Each mask is an int of 5001 bits ~ 625 bytes.
# Total memory ~ 60 MB. Acceptable.

residue_masks = {}
for p in PRIMES_LIMITED:
    masks = [0] * p
    for x in range(N + 1):
        r = x % p
        masks[r] |= (1 << x)
    residue_masks[p] = masks

def _popcount(x):
    return bin(x).count('1')

def _is_admissible_mask(mask, size):
    """Check if the set represented by mask is admissible."""
    for p in PRIMES_LIMITED:
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

def _greedy_bitset(seed=None, strategy=0):
    """
    Greedy deletion using bitsets.
    strategy 0: Smallest violating prime, min count residue
    strategy 1: Random violating prime, min count residue
    strategy 2: Random violating prime, random among min count
    strategy 3: Random violating prime, random among min count + 1
    """
    if seed is not None:
        random.seed(seed)
    
    mask = (1 << (N + 1)) - 1
    size = N + 1
    
    while True:
        # Find violating primes
        violating_primes = []
        for p in PRIMES_LIMITED:
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
            p = random.choice(violating_primes)
        
        masks = residue_masks[p]
        counts = []
        for r in range(p):
            counts.append(_popcount(mask & masks[r]))
        
        min_count = min(counts)
        
        if strategy == 0 or strategy == 1:
            # Min count, tie-break lowest or random
            candidates = [r for r in range(p) if counts[r] == min_count]
            if strategy == 0:
                r_to_remove = candidates[0]
            else:
                r_to_remove = random.choice(candidates)
        elif strategy == 2:
            # Random among min count
            candidates = [r for r in range(p) if counts[r] == min_count]
            r_to_remove = random.choice(candidates)
        elif strategy == 3:
            # Random among min count + 1
            candidates = [r for r in range(p) if counts[r] <= min_count + 1]
            r_to_remove = random.choice(candidates)
        else:
            r_to_remove = 0
            
        # Remove
        mask &= ~masks[r_to_remove]
        size -= counts[r_to_remove]
        
        if size == 0:
            break
            
    # Convert mask to list
    result = []
    m = mask
    i = 0
    while m:
        if m & 1:
            result.append(i)
        m >>= 1
        i += 1
    return result

def _local_improve_bitset(mask, size, max_iters=20):
    """
    Try to add back elements.
    We need to know which elements are removed.
    full_mask = (1 << (N+1)) - 1
    removed_mask = full_mask ^ mask
    """
    full_mask = (1 << (N + 1)) - 1
    removed_mask = full_mask ^ mask
    
    for _ in range(max_iters):
        # Pick a random element from removed_mask
        if removed_mask == 0:
            break
        # Get a random bit set in removed_mask
        # random.choice is slow on int bits.
        # We can extract bits into a list once? No, too slow.
        # Use random integer sampling?
        # Let's try: pick a random index, check if it's in removed_mask.
        # Or extract bits.
        
        # Efficient way to pick a random set bit:
        # random_val = random.getrandbits(N+1)
        # intersection = removed_mask & random_val
        # if intersection:
        #     # pick lowest set bit
        #     lsb = intersection & -intersection
        #     x = lsb.bit_length() - 1
        # else:
        #     continue
        
        found = False
        for _ in range(100): # Try a few times to find a set bit
            x = random.randint(0, N)
            if (removed_mask >> x) & 1:
                # Try to add x
                new_mask = mask | (1 << x)
                new_size = size + 1
                if _is_admissible_mask(new_mask, new_size):
                    mask = new_mask
                    size = new_size
                    removed_mask ^= (1 << x)
                    found = True
                    break
        if not found:
            # If we fail to find any addable element in this iteration, 
            # it might be that no element can be added, or we just missed.
            # To be safe, if we tried hard and failed, we might break.
            # But random sampling might miss.
            # For now, break if we fail a few times in a row?
            # Let's just break for simplicity, as adding back is hard.
            break
            
    # Convert mask to list
    result = []
    m = mask
    i = 0
    while m:
        if m & 1:
            result.append(i)
        m >>= 1
        i += 1
    return result

def construct():
    start_time = time.time()
    best_size = 0
    best_tuple = []
    
    # 1. Deterministic Bitset Greedy
    t0 = time.time()
    res = _greedy_bitset(seed=None, strategy=0)
    if len(res) > best_size:
        best_size = len(res)
        best_tuple = res
    
    # 2. Local Improvement (Bitset)
    if time.time() - start_time < MAX_TIME:
        # We need to pass the mask to local improve, but we only have list.
        # Let's reconstruct mask from best_tuple for local improve?
        # Or just run local improve on the bitset version directly in the loop?
        # Let's refactor: _greedy_bitset returns mask and size.
        
        # For now, let's just do more random restarts of the bitset greedy.
        pass
    
    # 3. Randomized Bitset Greedy
    for i in range(5000):
        if time.time() - start_time > MAX_TIME - 10:
            break
        
        strat = random.randint(0, 3)
        res = _greedy_bitset(seed=i, strategy=strat)
        if len(res) > best_size:
            best_size = len(res)
            best_tuple = res
            
    return best_tuple

# Refactor to return mask for better local improvement potential
def _greedy_bitset_internal(seed=None, strategy=0):
    if seed is not None:
        random.seed(seed)
    
    mask = (1 << (N + 1)) - 1
    size = N + 1
    
    while True:
        violating_primes = []
        for p in PRIMES_LIMITED:
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
            p = random.choice(violating_primes)
        
        masks = residue_masks[p]
        counts = []
        for r in range(p):
            counts.append(_popcount(mask & masks[r]))
        
        min_count = min(counts)
        
        if strategy == 0 or strategy == 1:
            candidates = [r for r in range(p) if counts[r] == min_count]
            if strategy == 0:
                r_to_remove = candidates[0]
            else:
                r_to_remove = random.choice(candidates)
        elif strategy == 2:
            candidates = [r for r in range(p) if counts[r] == min_count]
            r_to_remove = random.choice(candidates)
        elif strategy == 3:
            candidates = [r for r in range(p) if counts[r] <= min_count + 1]
            r_to_remove = random.choice(candidates)
        else:
            r_to_remove = 0
            
        mask &= ~masks[r_to_remove]
        size -= counts[r_to_remove]
        
        if size == 0:
            break
            
    return mask, size

def _mask_to_list(mask):
    result = []
    m = mask
    i = 0
    while m:
        if m & 1:
            result.append(i)
        m >>= 1
        i += 1
    return result

def construct():
    start_time = time.time()
    best_size = 0
    best_tuple = []
    
    # 1. Deterministic
    mask, size = _greedy_bitset_internal(seed=None, strategy=0)
    if size > best_size:
        best_size = size
        best_tuple = _mask_to_list(mask)
    
    # 2. Randomized
    for i in range(10000):
        if time.time() - start_time > MAX_TIME - 10:
            break
        
        strat = random.randint(0, 3)
        mask, size = _greedy_bitset_internal(seed=i, strategy=strat)
        if size > best_size:
            best_size = size
            best_tuple = _mask_to_list(mask)
            
    return best_tuple
