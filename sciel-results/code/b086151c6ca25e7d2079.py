"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000."""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 85.0

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor sieve up to 20000
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[i] == i:
        for j in range(i * i, _MAX_SUM + 1, i):
            if _sieve[j] == j:
                _sieve[j] = i

# Precompute set of prime factors for each number up to 20000
_pf_cache = [set() for _ in range(_MAX_SUM + 1)]
for i in range(2, _MAX_SUM + 1):
    n = i
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    _pf_cache[i] = factors

def _compute_score_and_counts(nums):
    """
    Compute the set of distinct prime factors and the count of each prime
    across all pairwise sums.
    Returns: (score, counts_dict)
    """
    n = len(nums)
    counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            for p in _pf_cache[s]:
                counts[p] = counts.get(p, 0) + 1
    return len(counts), counts

def construct():
    random.seed(42)
    
    best_score = float('inf')
    best_candidate = None
    best_counts = None

    # Initial candidates
    candidates = [
        [6 * i for i in range(1, 13)],
        [12 * i for i in range(1, 13)],
        [30 * i for i in range(1, 13)],
        [60 * i for i in range(1, 13)],
        [120 * i for i in range(1, 13)],
        [2 * i for i in range(1, 13)],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        [3**i for i in range(1, 13)],
    ]
    
    # Add some random multiples of 6
    for _ in range(10):
        cand = sorted(random.sample([6 * i for i in range(1, 1667)], 12))
        candidates.append(cand)

    # Evaluate initial candidates
    for cand in candidates:
        if _deadline_hit():
            break
        if len(set(cand)) != 12:
            continue
        if not all(1 <= x <= 10000 for x in cand):
            continue
        
        score, counts = _compute_score_and_counts(cand)
        if score < best_score:
            best_score = score
            best_candidate = cand[:]
            best_counts = counts

    if best_candidate is None:
        best_candidate = list(range(1, 13))
        best_score, best_counts = _compute_score_and_counts(best_candidate)

    current = best_candidate[:]
    current_counts = best_counts.copy()
    current_score = best_score
    n = 12

    # Local Search with Fast Incremental Updates
    # We will perform multiple restarts if time permits.
    
    def _try_move(idx, new_val):
        """
        Attempts to replace current[idx] with new_val.
        Returns True if the move was accepted (score improved or equal and diversity increased?), 
        actually we only accept if score improves.
        Modifies current, current_counts, current_score in place if accepted.
        """
        nonlocal current_score
        if new_val in current and current[idx] != new_val:
            return False
        if new_val == current[idx]:
            return True
        if not (1 <= new_val <= 10000):
            return False

        old_val = current[idx]
        
        # Calculate delta
        delta_score = 0
        # Temporarily store changes to revert if needed? 
        # Or just apply, check score, revert if worse.
        
        # Apply changes
        changes = [] # list of (prime, delta_count)
        
        for j in range(n):
            if j == idx:
                continue
            
            old_sum = old_val + current[j]
            new_sum = new_val + current[j]
            
            old_pf = _pf_cache[old_sum]
            new_pf = _pf_cache[new_sum]
            
            # Process old factors
            for p in old_pf:
                current_counts[p] -= 1
                if current_counts[p] == 0:
                    delta_score -= 1 # Prime removed from set
                changes.append((p, -1))
                
            # Process new factors
            for p in new_pf:
                if current_counts[p] == 0:
                    delta_score += 1 # Prime added to set
                current_counts[p] += 1
                changes.append((p, 1))
        
        new_score = current_score + delta_score
        
        if new_score < current_score:
            # Accept
            current[idx] = new_val
            current_score = new_score
            return True
        else:
            # Revert
            for p, delta in changes:
                current_counts[p] -= delta
            return False

    # Main search loop
    no_improve_steps = 0
    max_no_improve = 500
    
    while not _deadline_hit():
        if no_improve_steps > max_no_improve:
            # Random restart or large perturbation
            # Perturb current solution
            for _ in range(3):
                idx = random.randint(0, n - 1)
                # Try a random value
                new_val = random.randint(1, 10000)
                # Bias towards multiples of 6
                if random.random() < 0.5:
                    new_val = random.randint(1, 1666) * 6
                _try_move(idx, new_val)
            no_improve_steps = 0
            continue

        # Pick a random index to modify
        idx = random.randint(0, n - 1)
        old_val = current[idx]
        
        # Generate candidate values
        # 1. Small perturbations
        # 2. Multiples of 6, 12, 30
        # 3. Random
        
        tried = set()
        
        # Perturbations
        for d in range(-10, 11):
            v = old_val + d
            if 1 <= v <= 10000 and v not in tried:
                tried.add(v)
                if _try_move(idx, v):
                    no_improve_steps = 0
                    break
        else:
            # Multiples of 6
            for _ in range(10):
                v = random.randint(1, 1666) * 6
                if v not in tried:
                    tried.add(v)
                    if _try_move(idx, v):
                        no_improve_steps = 0
                        break
            else:
                # Random
                for _ in range(10):
                    v = random.randint(1, 10000)
                    if v not in tried:
                        tried.add(v)
                        if _try_move(idx, v):
                            no_improve_steps = 0
                            break
                else:
                    no_improve_steps += 1
        
        # Update best if improved
        if current_score < best_score:
            best_score = current_score
            best_candidate = current[:]
            best_counts = current_counts.copy()

    # Final verification
    result = best_candidate
    assert len(set(result)) == 12
    assert all(1 <= x <= 10000 for x in result)
    
    return result

# EVOLVE-BLOCK-END
