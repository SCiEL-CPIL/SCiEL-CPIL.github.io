"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000."""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 50.0

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor sieve up to 20000 (max sum is 2*10000)
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[i] == i:
        for j in range(i * i, _MAX_SUM + 1, i):
            if _sieve[j] == j:
                _sieve[j] = i

def _prime_factors_fast(n):
    """Return frozenset of prime factors of n using precomputed sieve."""
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return frozenset(factors)

def _compute_factor_sets(nums):
    """For each pair (i,j), compute frozenset of prime factors of nums[i]+nums[j]."""
    n = len(nums)
    pair_factors = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            pf = _prime_factors_fast(s)
            pair_factors[i][j] = pf
            pair_factors[j][i] = pf
    return pair_factors

def _total_factor_count(pair_factors):
    """Count distinct primes across all pair factor sets."""
    all_primes = set()
    n = len(pair_factors)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes |= pair_factors[i][j]
    return len(all_primes)

def _base_primes_excluding(pf, exclude_idx, n):
    """Get union of all pair factors excluding pairs involving exclude_idx."""
    base = set()
    for i in range(n):
        if i == exclude_idx:
            continue
        for j in range(i+1, n):
            if j == exclude_idx:
                continue
            base |= pf[i][j]
    return base

# Precompute smooth numbers (2,3,5,7-smooth) up to 20000
_smooth_numbers = []
for _a in range(15):
    _p2 = 1 << _a
    if _p2 > 20000:
        break
    for _b in range(12):
        _p3 = 3 ** _b
        if _p2 * _p3 > 20000:
            break
        for _c in range(9):
            _p5 = 5 ** _c
            if _p2 * _p3 * _p5 > 20000:
                break
            for _d in range(7):
                _p7 = 7 ** _d
                _smooth = _p2 * _p3 * _p5 * _p7
                if _smooth > 20000:
                    break
                _smooth_numbers.append(_smooth)
_smooth_numbers = sorted(set(_smooth_numbers))

# Precompute: for each smooth number s, which primes divide it
_smooth_prime_masks = {}
for s in _smooth_numbers:
    _smooth_prime_masks[s] = _prime_factors_fast(s)

# Precompute a good candidate pool for local search
_candidate_pool = set()
for s in _smooth_numbers[:500]:
    _candidate_pool.add(s)
for v in range(1, 300):
    _candidate_pool.add(v)
for p in [2, 3, 5, 6, 10, 12, 15, 20, 30, 60, 120]:
    for k in range(1, 100):
        val = p * k
        if val <= 10000:
            _candidate_pool.add(val)
_candidate_pool = sorted(_candidate_pool)

def _find_smooth_clique(target_primes, min_size=12):
    """Find a clique of size min_size in the graph where nodes are numbers 1..10000
    and edges connect numbers whose sum has only prime factors in target_primes."""
    tp_set = set(target_primes)
    
    # Use a small, curated set of candidates to ensure speed
    # Take the first 100 smooth numbers as the main candidate pool for clique search
    candidates = _smooth_numbers[:100]
    
    # Precompute compatibility matrix
    n_cands = len(candidates)
    compat = [[False] * n_cands for _ in range(n_cands)]
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            s = candidates[i] + candidates[j]
            if s <= _MAX_SUM:
                pf = _prime_factors_fast(s)
                if pf.issubset(tp_set):
                    compat[i][j] = True
                    compat[j][i] = True
    
    # Backtracking to find clique of size min_size
    best_clique = []
    
    def backtrack(start_idx, current_clique):
        nonlocal best_clique
        if len(current_clique) >= min_size:
            best_clique = current_clique[:]
            return True
        if _deadline_hit():
            return False
        
        for i in range(start_idx, n_cands):
            x_idx = i
            # Check compatibility with all in current_clique
            compatible = True
            for y_idx in current_clique:
                if not compat[x_idx][y_idx]:
                    compatible = False
                    break
            if not compatible:
                continue
            if backtrack(i + 1, current_clique + [x_idx]):
                return True
        return False
    
    if backtrack(0, []):
        return [candidates[i] for i in best_clique]
    return None

def construct():
    random.seed(42)
    
    # Generate initial candidates
    candidates = []
    
    # Strategy 1: Powers of 2
    candidates.append([2**i for i in range(1, 13)])
    
    # Strategy 2: Multiples of highly composite numbers
    for base in [30, 60, 120, 210, 360, 840, 2520]:
        candidates.append([base * (i+1) for i in range(12)])
    
    # Strategy 3: Small consecutive numbers
    candidates.append(list(range(1, 13)))
    candidates.append(list(range(2, 14)))
    candidates.append(list(range(10, 22)))
    
    # Strategy 4: Even numbers
    candidates.append(list(range(2, 25, 2)))
    candidates.append(list(range(10, 33, 2)))
    
    # Strategy 5: Numbers of form 2^a * 3^b
    cand_23 = []
    for a in range(15):
        for b in range(12):
            val = 2**a * 3**b
            if 1 <= val <= 10000 and val not in cand_23:
                cand_23.append(val)
            if len(cand_23) >= 12:
                break
        if len(cand_23) >= 12:
            break
    if len(cand_23) >= 12:
        candidates.append(cand_23[:12])
    
    # Strategy 6: Arithmetic progressions with lcm-based differences
    for d in [30, 60, 120, 210, 420]:
        for a in [1, 2, 3, 5, 10, 15, 30]:
            cand = [a + d * i for i in range(12)]
            if all(1 <= x <= 10000 for x in cand) and len(set(cand)) == 12:
                candidates.append(cand)
    
    # Strategy 7: Use smooth numbers as candidates
    if len(_smooth_numbers) >= 12:
        candidates.append(_smooth_numbers[:12])
        candidates.append(_smooth_numbers[5:17])
        candidates.append(_smooth_numbers[10:22])
        candidates.append(_smooth_numbers[15:27])
    
    # Strategy 8: Numbers that are multiples of small primes
    for p in [2, 3, 5, 6, 10, 12, 15, 20, 30]:
        cand = []
        for k in range(1, 100):
            val = p * k
            if 1 <= val <= 10000 and val not in cand:
                cand.append(val)
            if len(cand) >= 12:
                break
        if len(cand) == 12:
            candidates.append(cand)
    
    # Strategy 9: Smooth clique search with {2,3,5,7}
    if not _deadline_hit():
        smooth_clique = _find_smooth_clique({2, 3, 5, 7}, 12)
        if smooth_clique:
            candidates.append(smooth_clique)
    
    # Strategy 10: Smooth clique with {2,3,5}
    if not _deadline_hit():
        smooth_clique3 = _find_smooth_clique({2, 3, 5}, 12)
        if smooth_clique3:
            candidates.append(smooth_clique3)
    
    # Evaluate all candidates
    best_candidate = None
    best_score = float('inf')
    best_pf = None
    
    for cand in candidates:
        if _deadline_hit():
            break
        if len(set(cand)) != 12:
            continue
        if not all(1 <= x <= 10000 for x in cand):
            continue
        pf = _compute_factor_sets(cand)
        score = _total_factor_count(pf)
        if score < best_score:
            best_score = score
            best_candidate = cand[:]
            best_pf = pf
    
    if best_candidate is None:
        best_candidate = list(range(1, 13))
        best_pf = _compute_factor_sets(best_candidate)
        best_score = _total_factor_count(best_pf)
    
    # Local search
    current = best_candidate[:]
    current_pf = best_pf
    current_score = best_score
    n = 12
    
    # Precompute base primes for each index
    base_primes_cache = [_base_primes_excluding(current_pf, i, n) for i in range(n)]
    
    # Build a targeted candidate pool for local search
    # Include smooth numbers, small numbers, and values derived from smooth - current[j]
    local_pool = set()
    for s in _smooth_numbers[:400]:
        local_pool.add(s)
    for v in range(1, 250):
        local_pool.add(v)
    for p in [2, 3, 5, 6, 10, 12, 15, 20, 30, 60]:
        for k in range(1, 80):
            val = p * k
            if val <= 10000:
                local_pool.add(val)
    # Add values that make smooth sums with current elements
    for j in range(n):
        for s in _smooth_numbers[:200]:
            v = s - current[j]
            if 1 <= v <= 10000:
                local_pool.add(v)
    local_pool = sorted(local_pool)
    
    # Main local search loop
    no_improve_count = 0
    
    while not _deadline_hit():
        improved = False
        
        for idx in range(n):
            if _deadline_hit():
                break
            
            old_val = current[idx]
            base_primes = base_primes_cache[idx]
            
            best_new_val = None
            best_new_score = current_score
            best_new_pf = None
            best_new_primes = None
            
            # Build candidate set for this index
            cand_vals = set()
            
            # Values near old value
            for d in range(-50, 51):
                v = old_val + d
                if 1 <= v <= 10000:
                    cand_vals.add(v)
            
            # Values from local pool that are promising
            for v in local_pool:
                if v in current:
                    continue
                # Quick check: does v + current[j] introduce new primes?
                # Only add if it doesn't immediately make things worse
                introduces_new = False
                for j in range(n):
                    if j == idx:
                        continue
                    s = v + current[j]
                    pf = _prime_factors_fast(s)
                    if not pf.issubset(base_primes) and len(pf - base_primes) > 1:
                        introduces_new = True
                        break
                if not introduces_new:
                    cand_vals.add(v)
            
            # Remove values already in current
            current_set = set(current)
            cand_vals -= current_set
            
            # Limit candidates to keep evaluation fast
            if len(cand_vals) > 50:
                # Prioritize: smooth numbers first, then near old_val, then rest
                smooth_cands = [v for v in cand_vals if v in _smooth_prime_masks]
                near_cands = [v for v in cand_vals if abs(v - old_val) <= 20 and v not in smooth_cands]
                rest = [v for v in cand_vals if v not in smooth_cands and v not in near_cands]
                # Take top 20 smooth, top 10 near, top 10 rest
                cand_vals = set(smooth_cands[:20]) | set(near_cands[:10]) | set(rest[:10])
            
            # Evaluate candidates
            for new_val in cand_vals:
                if new_val in current:
                    continue
                
                new_primes = set(base_primes)
                for j in range(n):
                    if j == idx:
                        continue
                    s = new_val + current[j]
                    pf = _prime_factors_fast(s)
                    new_primes |= pf
                
                new_score = len(new_primes)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_val = new_val
                    best_new_primes = new_primes
                    # Defer pf update until we confirm this is the best
                    best_new_pf = None  # will be computed after loop
            
            if best_new_val is not None and best_new_score < current_score:
                # Update current
                current[idx] = best_new_val
                # Recompute only the pairs involving idx
                for j in range(n):
                    if j != idx:
                        s = current[idx] + current[j]
                        pf = _prime_factors_fast(s)
                        current_pf[idx][j] = pf
                        current_pf[j][idx] = pf
                current_score = best_new_score
                # Update base_primes_cache only for affected indices
                # This is O(N^2) but N=12 is small. 
                # However, doing it every improvement might be costly if many improvements happen.
                # We can just recompute the score directly in the next iteration if we want, 
                # but base_primes_cache is used for quick rejection.
                # Let's just update the specific ones that changed relative to idx.
                # Actually, base_primes_cache[i] is the union of all pairs NOT involving i.
                # If we change idx, then for any j != idx, the pair (j, idx) changes.
                # So base_primes_cache[j] changes for all j != idx.
                # And base_primes_cache[idx] changes because the pairs involving idx changed.
                # So we must update all.
                for i in range(n):
                    base_primes_cache[i] = _base_primes_excluding(current_pf, i, n)
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            # Try random perturbation to escape local minimum
            if no_improve_count >= 3 and not _deadline_hit():
                # Limit perturbation attempts to avoid wasting time
                for _ in range(5):
                    if _deadline_hit():
                        break
                    new_cand = current[:]
                    num_changes = random.randint(1, 2)
                    for _ in range(num_changes):
                        idx = random.randint(0, 11)
                        new_cand[idx] = random.choice(local_pool)
                    
                    if len(set(new_cand)) == 12 and all(1 <= x <= 10000 for x in new_cand):
                        new_pf = _compute_factor_sets(new_cand)
                        new_score = _total_factor_count(new_pf)
                        if new_score < current_score:
                            current = new_cand
                            current_pf = new_pf
                            current_score = new_score
                            base_primes_cache = [_base_primes_excluding(current_pf, i, n) for i in range(n)]
                            no_improve_count = 0
                            break
    
    # Final verification
    result = current
    if len(set(result)) != 12 or not all(1 <= x <= 10000 for x in result):
        result = [360 * (i+1) for i in range(12)]
    
    return result
