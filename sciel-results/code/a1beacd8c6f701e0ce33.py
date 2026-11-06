"""
Strategy: Deterministic Rational Geometric Progression (RGP) Search + Collision-Aware Local Search.
This approach models the set as a truncated geometric progression with a rational ratio p/q.
This structure is chosen because it creates multiplicative collisions (e.g., a*p/q * b*p/q = a*b*(p/q)^2)
which reduce the size of A*A. We grid-search over small integers p, q to find the best base structure,
then refine with a local search that specifically targets additive collisions by swapping elements
to create symmetric pairs (x, y) such that x+y equals an existing sum, thereby reducing |A+A|.
"""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from itertools import product

_N = 20
_MAX_VAL = 10000
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin

def _score(a):
    """Return max(|A+A|, |A*A|) for sorted distinct positive integers a."""
    if not a:
        return 10**9
    a_arr = np.asarray(a, dtype=np.int64)
    if len(a_arr) != _N or np.any(a_arr < 1) or np.any(a_arr > _MAX_VAL):
        return 10**9
    if len(set(a_arr.tolist())) != _N:
        return 10**9
    
    # Compute sumset and product set sizes
    # Using broadcasting for efficiency
    s = a_arr[:, None] + a_arr[None, :]
    p = a_arr[:, None] * a_arr[None, :]
    
    # Count unique elements
    # np.unique is efficient for these sizes (20x20=400 elements)
    n_sums = len(np.unique(s))
    n_prods = len(np.unique(p))
    
    return max(n_sums, n_prods)

def _is_valid(a):
    """Check if a is a valid set of 20 distinct positive integers <= 10000."""
    if len(a) != _N:
        return False
    if len(set(a)) != _N:
        return False
    return all(isinstance(x, int) and 1 <= x <= _MAX_VAL for x in a)

def _generate_rgp(p, q, a0):
    """
    Generate a set from a rational geometric progression a0 * (p/q)^k.
    We round to nearest integer. To ensure distinctness and bounds, we clip and deduplicate.
    If we lose elements due to collisions or bounds, we return None.
    """
    vals = []
    current = a0
    for k in range(_N):
        # Calculate a0 * (p/q)^k
        # To avoid floating point drift, we can compute as a0 * p^k / q^k
        # But for k up to 19, p^k can be large. 
        # Using float is usually fine for this scale, but let's be careful.
        val = a0 * (p ** k) / (q ** k)
        iv = int(round(val))
        if iv < 1:
            iv = 1
        if iv > _MAX_VAL:
            # If we exceed max, we can't add more distinct values easily without breaking GP structure
            # However, the task requires 20 distinct values <= 10000.
            # If the GP exceeds 10000, it's invalid for this specific parameter set.
            return None
        vals.append(iv)
    
    # Sort and check distinctness
    vals = sorted(vals)
    if len(set(vals)) != _N:
        return None
    
    return vals

def _local_search_collision_aware(a, max_iter=1000):
    """
    Local search that tries to swap elements to create additive collisions.
    Specifically, it looks for pairs (i, j) and (k, l) such that a[i]+a[j] == a[k]+a[l] 
    by swapping a[j] with a candidate value.
    """
    a = np.array(a, dtype=np.int64)
    best_score = _score(a)
    a_set = set(a.tolist())
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(_N))
        random.shuffle(indices)
        
        for i in indices:
            orig_val = a[i]
            # Try changing a[i] to a nearby value
            # We focus on values that might create collisions with existing sums
            # Heuristic: try values in [1, 10000] that are close to current value
            # Or specifically, try to make a[i] + a[j] match a[k] + a[l]
            
            # A simpler effective strategy: try all values in a small range around orig_val
            # And also try values that are "sum complements" of other pairs
            candidates = set()
            
            # 1. Local perturbations
            for delta in range(-10, 11):
                nv = orig_val + delta
                if 1 <= nv <= _MAX_VAL and nv != orig_val:
                    candidates.add(nv)
            
            # 2. Structural candidates: 
            # For each pair (j, k), we want a[i] + a[j] == a[k] + a[l] for some l.
            # This is complex. Instead, let's just try to minimize the score via random valid swaps
            # but prioritize swaps that keep the set "dense" or "structured".
            
            for nv in candidates:
                if nv in a_set:
                    continue
                if nv == orig_val:
                    continue
                    
                # Apply swap
                a[i] = nv
                new_score = _score(a)
                
                if new_score < best_score:
                    best_score = new_score
                    a_set.remove(orig_val)
                    a_set.add(nv)
                    improved = True
                    break
                else:
                    # Revert
                    a[i] = orig_val
            
            if improved:
                break
        
        if not improved:
            # If no improvement in local perturbations, try a larger jump
            # Pick a random element and replace it with a random value from 1..10000
            # This helps escape local minima
            i = random.randint(0, _N - 1)
            orig_val = a[i]
            for _ in range(20): # Try 20 random replacements
                nv = random.randint(1, _MAX_VAL)
                if nv in a_set or nv == orig_val:
                    continue
                a[i] = nv
                new_score = _score(a)
                if new_score < best_score:
                    best_score = new_score
                    a_set.remove(orig_val)
                    a_set.add(nv)
                    improved = True
                    break
                else:
                    a[i] = orig_val
            if not improved:
                # Accept a worse move with low probability (Simulated Annealing style)
                if random.random() < 0.05:
                    i = random.randint(0, _N - 1)
                    orig_val = a[i]
                    nv = random.randint(1, _MAX_VAL)
                    if nv in a_set or nv == orig_val:
                        continue
                    a[i] = nv
                    new_score = _score(a)
                    a_set.remove(orig_val)
                    a_set.add(nv)
                    # Don't update best_score, just accept the state
                    improved = True
                    break
        
        if not improved:
            break
            
    return a.tolist(), best_score

def construct():
    start_time = time.time()
    best_set = None
    best_score = 10**9
    
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Deterministic Grid Search over Rational Geometric Progressions
    # We search for p, q such that p/q is close to 1 (to keep values in range) 
    # or such that p/q creates good multiplicative collisions.
    # Common small ratios: 3/2, 4/3, 5/4, etc.
    
    p_range = range(1, 15)
    q_range = range(1, 15)
    
    # Pre-filter valid (p,q) pairs where p != q to avoid constant sequences
    valid_pairs = [(p, q) for p in p_range for q in q_range if p != q]
    
    # Sort by "closeness" to 1 first, as these are most likely to fit in [1, 10000]
    # and have many terms.
    valid_pairs.sort(key=lambda x: abs(x[0]/x[1] - 1.0))
    
    # Limit the number of pairs to check to respect time budget
    # 100 pairs * 20 terms * scoring cost is fast.
    max_pairs_to_check = 200
    
    for p, q in valid_pairs[:max_pairs_to_check]:
        if time.time() > _DEADLINE - 20:
            break
            
        # Determine valid range for a0
        # a0 * (p/q)^(N-1) <= 10000
        # a0 <= 10000 * (q/p)^(N-1)
        # a0 >= 1
        ratio = p / q
        if ratio > 1:
            max_a0 = int(10000 / (ratio ** (_N - 1)))
            if max_a0 < 1:
                continue
            a0_max = max_a0
        else:
            # If ratio < 1, the sequence decreases.
            # a0 must be <= 10000.
            # The smallest term is a0 * (p/q)^(N-1) >= 1 => a0 >= (q/p)^(N-1)
            min_a0 = int(np.ceil((1/ratio) ** (_N - 1)))
            if min_a0 > 10000:
                continue
            a0_max = 10000
            
        # Sample a0 values
        # We want to try a0 values that result in "nice" integers.
        # If p/q is a simple fraction, a0 being a multiple of q^(N-1) might help, 
        # but we are rounding anyway.
        
        # Try a few specific a0 values:
        # 1. Minimum valid a0
        # 2. Maximum valid a0
        # 3. Some intermediate values
        
        candidates_a0 = set()
        if ratio > 1:
            candidates_a0.add(1)
            candidates_a0.add(a0_max)
            # Add some logarithmic steps
            for exp in range(0, 4):
                val = int(10 ** exp)
                if 1 <= val <= a0_max:
                    candidates_a0.add(val)
        else:
            candidates_a0.add(min_a0)
            candidates_a0.add(10000)
            for exp in range(0, 5):
                val = int(10 ** exp)
                if min_a0 <= val <= 10000:
                    candidates_a0.add(val)
        
        for a0 in candidates_a0:
            if time.time() > _DEADLINE - 20:
                break
            vals = _generate_rgp(p, q, a0)
            if vals is None:
                continue
            
            sc = _score(vals)
            if sc < best_score:
                best_score = sc
                best_set = vals
                
    # Phase 2: Local Search Refinement
    if best_set is not None:
        polished, sc = _local_search_collision_aware(best_set, max_iter=500)
        if sc < best_score:
            best_score = sc
            best_set = polished
    else:
        # Fallback: Random search if GP search failed (unlikely)
        for _ in range(100):
            if time.time() > _DEADLINE - 5:
                break
            candidate = sorted(random.sample(range(1, _MAX_VAL + 1), _N))
            sc = _score(candidate)
            if sc < best_score:
                best_score = sc
                best_set = candidate
        
        if best_set:
            polished, sc = _local_search_collision_aware(best_set, max_iter=500)
            if sc < best_score:
                best_score = sc
                best_set = polished

    # Final Fallback
    if best_set is None:
        best_set = list(range(1, 21))

    return best_set
# EVOLVE-BLOCK-END
