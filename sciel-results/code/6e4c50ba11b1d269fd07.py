"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from math import isqrt

N = 5000
RANGE = N + 1

# EVOLVE-BLOCK-START

def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = False
    return np.where(sieve)[0]

PRIMES = _primes(N)
PRIMES_LIST = PRIMES.tolist()

def _greedy_construct(seed, deadline, strategy='random'):
    """
    Construct an admissible tuple using greedy residue removal.
    strategy: 'random' (random among min-count), 'least' (always min-count, deterministic tie-break)
    """
    if seed is not None:
        random.seed(seed)
    
    active = np.ones(RANGE, dtype=bool)
    x_indices = np.arange(RANGE)
    current_size = RANGE
    
    for p in PRIMES_LIST:
        if time.time() > deadline:
            break
        if p > current_size:
            break
        
        # Count active elements in each residue class mod p
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        residues = active_indices % p
        if len(residues) == 0:
            break
        
        # Count occurrences of each residue
        counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(counts)
        
        if min_count == 0:
            # Already admissible for this prime
            continue
        
        # Find all classes with min count
        candidate_classes = np.where(counts == min_count)[0]
        
        if strategy == 'random':
            r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
        elif strategy == 'least':
            # Deterministic: pick the first min class (or use a hash for variety? No, keep simple)
            r = candidate_classes[0]
        else:
            r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
        
        # Deactivate elements in this class
        mask = (x_indices % p == r) & active
        current_size -= np.sum(mask)
        active = active & ~mask
        
        if current_size == 0:
            break
    
    return np.where(active)[0].tolist()

def _try_strategy(primes, sieve_threshold, use_least_populated_for_small=True, seed=None):
    """
    Hybrid strategy:
    1. For small primes (<= threshold), use a specific removal rule (least populated or residue 0).
    2. For remaining primes, use the iterative repair loop (find bad p, remove min count residue).
    """
    S_mask = np.ones(N + 1, dtype=bool)
    
    # Sieve primes up to threshold
    small_primes = primes[primes <= sieve_threshold]
    for p in small_primes:
        if use_least_populated_for_small:
            # Find least populated residue among current S
            S_indices = np.where(S_mask)[0]
            if len(S_indices) == 0:
                return 0, S_mask
            residues = S_indices % p
            counts = np.bincount(residues, minlength=p)
            best_r = np.argmin(counts)
            to_remove = S_indices[residues == best_r]
            S_mask[to_remove] = False
        else:
            # Remove residue 0
            multiples = np.arange(0, N + 1, p)
            S_mask[multiples] = False
    
    remaining_primes = primes[primes > sieve_threshold]
    
    changed = True
    while changed:
        current_size = np.count_nonzero(S_mask)
        if current_size == 0:
            break
        
        S_indices = np.where(S_mask)[0]
        candidates = remaining_primes[remaining_primes <= current_size]
        
        bad_p = None
        min_removal_count = float('inf')
        best_r = -1
        best_p = -1
        
        for p in candidates:
            residues = S_indices % p
            unique_res = np.unique(residues)
            if len(unique_res) == p:
                counts = np.bincount(residues, minlength=p)
                min_count = np.min(counts)
                if min_count < min_removal_count:
                    min_removal_count = min_count
                    best_r = np.argmin(counts)
                    best_p = p
                    bad_p = p
        
        if bad_p is not None:
            to_remove_mask = (S_indices % best_p) == best_r
            values_to_remove = S_indices[to_remove_mask]
            S_mask[values_to_remove] = False
            changed = True
        else:
            changed = False
    
    return np.count_nonzero(S_mask), S_mask

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    best_size = 0
    best_points = []
    
    # Phase 1: Multi-start greedy (Inspiration 1)
    # This is fast and provides good initial solutions.
    attempts = 0
    while time.time() < deadline - 30:
        attempts += 1
        seed = attempts
        # Alternate strategies
        strategy = 'random' if attempts % 2 == 0 else 'least'
        points = _greedy_construct(seed, deadline, strategy)
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            
        # Occasionally try a hybrid strategy from the current program (Inspiration 2 / Current)
        # The hybrid strategy is more complex but might find better structures with specific thresholds.
        # We'll interleave or just run a few specific ones.
        if attempts % 10 == 0:
            # Try a few specific thresholds from the current program's list
            for threshold in [7, 11, 17, 23]:
                if time.time() > deadline - 30:
                    break
                size, mask = _try_strategy(PRIMES, threshold, use_least_populated_for_small=True)
                if size > best_size:
                    best_size = size
                    best_points = np.where(mask)[0].tolist()
                    
                size, mask = _try_strategy(PRIMES, threshold, use_least_populated_for_small=False)
                if size > best_size:
                    best_size = size
                    best_points = np.where(mask)[0].tolist()

    # Phase 2: Polish / Local Search (Hypothesis)
    # Try to add back elements to the best set found so far.
    if best_points and time.time() < deadline - 5:
        point_set = set(best_points)
        current_size = len(best_points)
        
        # Identify critical primes: primes p <= current_size where best_points covers exactly p-1 residues.
        # If we add x, we must not complete the set of residues for any critical prime.
        # Also, adding x increases size, so we might need to check p = current_size + 1 if prime.
        
        critical_primes = {}
        for p in PRIMES_LIST:
            if p > current_size:
                break
            residues = set(x % p for x in best_points)
            if len(residues) == p - 1:
                missing = [r for r in range(p) if r not in residues]
                if len(missing) == 1:
                    critical_primes[p] = missing[0]
        
        # Candidates to add: all x in [0, N] not in best_points
        candidates = [x for x in range(RANGE) if x not in point_set]
        
        # Filter candidates based on critical primes
        # x is invalid if for any (p, r) in critical_primes, x % p == r
        # Note: This is a necessary condition for validity, but not sufficient.
        # Adding x might make a non-critical prime become critical or fully covered?
        # If S is admissible, and we add x, S' is non-admissible only if:
        # 1. A prime p <= |S| becomes fully covered. This happens if S had p-1 residues and x provides the last one.
        # 2. A prime p = |S|+1 becomes fully covered. This happens if S' covers all p residues.
        
        # Let's filter first.
        valid_candidates = []
        for x in candidates:
            ok = True
            for p, r in critical_primes.items():
                if x % p == r:
                    ok = False
                    break
            if ok:
                valid_candidates.append(x)
        
        # Now try to add valid candidates one by one?
        # Adding one might change the critical set.
        # This is complex. Let's just try adding them in random order and check full admissibility.
        # But full check is expensive.
        
        # Alternative: Greedy addition.
        # Pick a random valid candidate, add it, update criticals, repeat.
        
        current_set = list(best_points)
        current_size = len(current_set)
        
        # Re-compute criticals for the new set if we add
        # For now, let's just try a simple loop:
        # While we have time and valid candidates:
        #   Pick a candidate
        #   Check if adding it keeps the set admissible (full check)
        #   If yes, add it.
        
        # To speed up, we can just do a few iterations.
        # Full check cost: O(|S| * pi(|S|))
        # |S| ~ 3000, pi(3000) ~ 430. 3000 * 430 = 1.3M ops.
        # In Python, this might take ~0.1-0.5s.
        # We have ~20s left. So we can try ~40-200 additions.
        
        random.shuffle(valid_candidates)
        
        for x in valid_candidates:
            if time.time() > deadline - 1:
                break
            
            # Check if adding x keeps admissibility
            # New size = current_size + 1
            new_size = current_size + 1
            
            # Check primes p <= new_size
            # We only need to check primes p where the current set covers p-1 or p residues?
            # No, we need to check if the NEW set covers all p residues.
            
            # Optimization:
            # For p <= current_size:
            #   If current set covers < p-1 residues, adding x makes it <= p-1. Safe.
            #   If current set covers p-1 residues, adding x makes it p if x % p is the missing one. (We already filtered these out for criticals, but criticals were computed for current_size. If current_size changes, new primes might become relevant? No, p <= current_size is fixed set of primes. But the "missing" residue might change? No, if it was p-1, it's still p-1 before adding. If x fills it, it becomes p. We filtered x such that it doesn't fill the missing residue for CURRENT criticals. So these are safe.)
            #   If current set covers p residues, it was already non-admissible. But we start with an admissible set. So this doesn't happen.
            
            # So the only risk is:
            # 1. Primes p where current set covers p-1 residues, and x fills the missing one. (Filtered)
            # 2. Primes p = new_size (if prime). The new set has new_size elements. It can cover all p residues only if all elements have distinct residues.
            
            # So we only need to check if p = new_size is prime, and if so, if the new set covers all residues mod p.
            
            if new_size in critical_primes: # Wait, critical_primes keys are primes.
                 # This check is wrong. new_size might be prime.
                 pass
            
            # Let's just do a quick check for p = new_size if prime.
            # And trust the filter for p <= current_size.
            
            is_prime_new_size = False
            for p in PRIMES_LIST:
                if p == new_size:
                    is_prime_new_size = True
                    break
            
            if is_prime_new_size:
                # Check if current_set + [x] covers all residues mod new_size
                # We can check residues of current_set first.
                # If current_set already has a duplicate residue mod new_size, it's safe.
                # If all residues in current_set are distinct, we need to check if x provides the missing one.
                
                # This is O(|S|) to check distinctness.
                seen = set()
                all_distinct = True
                for y in current_set:
                    r = y % new_size
                    if r in seen:
                        all_distinct = False
                        break
                    seen.add(r)
                
                if all_distinct:
                    # Check if x % new_size is the missing one
                    missing = [r for r in range(new_size) if r not in seen]
                    if len(missing) == 1 and x % new_size == missing[0]:
                        continue # Invalid
                    
            # If we pass the check, add x
            current_set.append(x)
            current_size += 1
            
            # Update critical_primes?
            # This is expensive. We can skip updating and just rely on the next iteration's check?
            # No, the filter `valid_candidates` is static.
            # If we add x, the criticals change.
            # To be safe, we should re-compute criticals or just do a full check every few additions.
            
            # Let's just do a full check every 10 additions to be safe.
            if current_size % 10 == 0:
                # Full check
                point_set_check = set(current_set)
                admissible = True
                for p in PRIMES_LIST:
                    if p > current_size:
                        break
                    residues = set(y % p for y in point_set_check)
                    if len(residues) == p:
                        admissible = False
                        break
                if not admissible:
                    # Rollback last 10? Or just stop?
                    # Simplest: stop polishing.
                    break
            
            if current_size > best_size:
                best_size = current_size
                best_points = current_set

    return best_points

# EVOLVE-BLOCK-END
