<<<<<<< SEARCH
"""Improved program: fast iterative random greedy restarts with sorted-list collision detection for dissociated reciprocal sets."""
import time
import random
from bisect import bisect_left

# EVOLVE-BLOCK-START

# LCM of 1..22 is 232792560
_LCM = 232792560
_INV_INT = {x: _LCM // x for x in range(1, 23)}

def _build_sorted_sums(elements):
    """
    Build sorted list of all subset sums for elements.
    Returns None if a collision is detected.
    """
    if not elements:
        return [0]
    sums = [0]
    for x in elements:
        inv = _INV_INT[x]
        new_sums = []
        for s in sums:
            new_sums.append(s + inv)
        # Merge sums and new_sums, checking for duplicates
        merged = []
        i, j = 0, 0
        while i < len(sums) or j < len(new_sums):
            if j >= len(new_sums) or (i < len(sums) and sums[i] < new_sums[j]):
                val = sums[i]
                i += 1
            elif i >= len(new_sums) or new_sums[j] < sums[i]:
                val = new_sums[j]
                j += 1
            else:
                # sums[i] == new_sums[j] -> collision
                return None
            if merged and merged[-1] == val:
                return None
            merged.append(val)
        sums = merged
    return sums

def _is_dissociated_fast(elements):
    """
    Fast check: can we add element to current set while maintaining dissociation?
    Uses sorted list approach.
    """
    return _build_sorted_sums(elements) is not None

def _can_add(current_sums, x):
    """
    Check if element x can be added to a set with current_sums (sorted list of all subset sums).
    Returns new sorted sums list, or None if collision.
    """
    inv = _INV_INT[x]
    new_sums = []
    for s in current_sums:
        new_sums.append(s + inv)
    # Merge current_sums and new_sums, checking for duplicates
    merged = []
    i, j = 0, 0
    while i < len(current_sums) or j < len(new_sums):
        if j >= len(new_sums) or (i < len(current_sums) and current_sums[i] < new_sums[j]):
            val = current_sums[i]
            i += 1
        elif i >= len(new_sums) or new_sums[j] < current_sums[i]:
            val = new_sums[j]
            j += 1
        else:
            # Collision
            return None
        if merged and merged[-1] == val:
            return None
        merged.append(val)
    return merged
=======
"""Improved program: fast iterative random greedy restarts with CRT-encoded NumPy collision detection for dissociated reciprocal sets."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# LCM of 1..22 is 232792560
_LCM = 232792560
_INV_INT = {x: _LCM // x for x in range(1, 23)}

# CRT moduli: use two large primes for collision detection
# We need moduli where the product exceeds max possible sum
# Max sum = sum of all reciprocals * LCM = LCM * sum(1/i for i=1..22)
# This is roughly LCM * 3.6 ≈ 8.4e8, so two 32-bit primes suffice
_MOD1 = 2147483647  # 2^31 - 1, prime
_MOD2 = 2147483629  # Another large prime

def _compute_crt_sums(elements):
    """
    Compute all subset sums modulo _MOD1 and _MOD2 using vectorized NumPy.
    Returns (sums_mod1, sums_mod2) as numpy arrays, or None if collision detected.
    """
    if not elements:
        return np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)
    
    sums_mod1 = np.array([0], dtype=np.int64)
    sums_mod2 = np.array([0], dtype=np.int64)
    
    for x in elements:
        inv = _INV_INT[x]
        inv_mod1 = inv % _MOD1
        inv_mod2 = inv % _MOD2
        
        # Vectorized: compute new sums
        new_sums_mod1 = (sums_mod1 + inv_mod1) % _MOD1
        new_sums_mod2 = (sums_mod2 + inv_mod2) % _MOD2
        
        # Concatenate
        sums_mod1 = np.concatenate([sums_mod1, new_sums_mod1])
        sums_mod2 = np.concatenate([sums_mod2, new_sums_mod2])
        
        # Check for collisions using set (unique check)
        # For efficiency, check periodically or use a hash
        n = len(sums_mod1)
        if n > 1000:
            # Combine into a single value for uniqueness check
            combined = (sums_mod1.astype(np.int64) * _MOD2 + sums_mod2).astype(np.uint64)
            if len(np.unique(combined)) < n:
                return None
    
    # Final collision check
    combined = (sums_mod1.astype(np.int64) * _MOD2 + sums_mod2).astype(np.uint64)
    if len(np.unique(combined)) < len(sums_mod1):
        return None
    
    return sums_mod1, sums_mod2

def _can_add_crt(sums_mod1, sums_mod2, x):
    """
    Check if element x can be added to a set with current CRT-encoded sums.
    Returns new (sums_mod1, sums_mod2), or None if collision.
    """
    inv = _INV_INT[x]
    inv_mod1 = inv % _MOD1
    inv_mod2 = inv % _MOD2
    
    new_sums_mod1 = (sums_mod1 + inv_mod1) % _MOD1
    new_sums_mod2 = (sums_mod2 + inv_mod2) % _MOD2
    
    sums_mod1_new = np.concatenate([sums_mod1, new_sums_mod1])
    sums_mod2_new = np.concatenate([sums_mod2, new_sums_mod2])
    
    # Collision check
    n = len(sums_mod1_new)
    combined = (sums_mod1_new.astype(np.int64) * _MOD2 + sums_mod2_new).astype(np.uint64)
    if len(np.unique(combined)) < n:
        return None
    
    return sums_mod1_new, sums_mod2_new
>>>>>>> REPLACE

