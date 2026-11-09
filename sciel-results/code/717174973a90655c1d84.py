"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _subset_sums(nums):
    """Generate all non-empty subset sums as a set."""
    sums = set()
    for x in nums:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        sums |= new_sums
    return sums

def _check_validity_fast(nums):
    """Fast validity check using set membership."""
    n = len(nums)
    if n == 0:
        return True
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    sums_set = sums
    max_sum = max(sums_set)
    for s in sums_set:
        if s == 0:
            continue
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return False
    return True

def _check_validity(nums):
    """Check divisibility-incomparable subset sum constraint."""
    return _check_validity_fast(nums)

def _score_candidate(num, current_sums):
    """Score a candidate: prefer numbers that don't create divisibility issues."""
    # Check if num itself divides or is divided by existing sums
    penalty = 0
    for s in current_sums:
        if s % num == 0:
            penalty += 1
        if num % s == 0:
            penalty += 1
    # Also check new sums created by adding num
    for s in current_sums:
        new_s = s + num
        for t in current_sums:
            if t != 0 and new_s % t == 0:
                penalty += 1
            if new_s % num == 0:
                penalty += 1
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best = [31, 32, 34]
    
    # Candidate pool: focus on larger numbers
    all_candidates = list(range(1, 257))
    
    # Strategy: Multiple rounds of greedy construction with different orderings
    # Key insight: larger numbers have fewer divisors in the sum range
    
    # Precompute "goodness" of each number (fewer small factors = better)
    def count_small_factors(n):
        count = 0
        for d in range(2, min(20, n) + 1):
            if n % d == 0:
                count += 1
        return count
    
    # Sort candidates: prefer larger numbers with fewer small factors
    def candidate_sort_key(x):
        return (-x, count_small_factors(x))
    
    sorted_candidates = sorted(all_candidates, key=candidate_sort_key)
    
    # Phase 1: Greedy construction with multiple random perturbations
    for phase in range(5):
        if time.time() > deadline - 15:
            break
        
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            
            rng = random.Random(trial * 104729 + phase * 7919)
            
            # Create a perturbed ordering
            if phase == 0:
                order = sorted_candidates[:]
            elif phase == 1:
                # Only top 150 candidates
                order = sorted_candidates[:150]
                rng.shuffle(order)
            elif phase == 2:
                # Random from top 200
                order = sorted_candidates[:200]
                rng.shuffle(order)
            elif phase == 3:
                # Mix: half sorted, half random
                top_half = sorted_candidates[:128]
                rest = sorted_candidates[128:]
                rng.shuffle(rest)
                order = top_half + rest
            else:
                # Fully random
                order = all_candidates[:]
                rng.shuffle(order)
            
            current = []
            current_sums = set()
            
            for num in order:
                if num in current:
                    continue
                if len(current) >= 10:
                    break
                
                # Quick check: does num alone conflict?
                trial_set = current + [num]
                
                # Fast pre-check before full validation
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + num)
                new_sums.add(num)
                
                # Check for duplicate sums
                if len(new_sums) != len(current_sums) + len([s for s in current_sums if s + num not in current_sums]) + 1:
                    continue
                
                # Full check
                if _check_validity_fast(trial_set):
                    current = trial_set
                    current_sums = new_sums
            
            if len(current) > len(best):
                best = current[:]
    
    # Phase 2: Extension attempts on best
    if time.time() < deadline - 10:
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            extended = best[:]
            extended_sums = _subset_sums(extended)
            
            # Try adding candidates in order of "least conflict"
            for num in sorted(all_candidates, key=lambda x: (-x, count_small_factors(x))):
                if num in extended:
                    continue
                if len(extended) >= 12:
                    break
                
                trial_set = extended + [num]
                if _check_validity_fast(trial_set):
                    extended = trial_set
                    extended_sums = _subset_sums(extended)
            
            if len(extended) > len(best):
                best = extended[:]
    
    # Phase 3: Local search - try replacing elements to enable extension
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            improved = False
            new_best = best[:]
            
            for i in range(len(new_best)):
                # Try replacing element i with something smaller/different
                candidates_to_try = [x for x in all_candidates if x not in new_best]
                rng = random.Random(hash((tuple(new_best), i)) % (2**32))
                rng.shuffle(candidates_to_try)
                
                for replacement in candidates_to_try[:50]:
                    trial = new_best[:]
                    trial[i] = replacement
                    if _check_validity_fast(trial):
                        new_best = trial
                        improved = True
                        break
                if improved:
                    break
            
            if improved and len(new_best) > len(best):
                best = new_best[:]
    
    # Phase 4: Final extension push
    if time.time() < deadline - 3:
        final = best[:]
        final_sums = _subset_sums(final)
        
        for num in range(256, 0, -1):
            if num in final:
                continue
            if len(final) >= 12:
                break
            
            trial_set = final + [num]
            if _check_validity_fast(trial_set):
                final = trial_set
                final_sums = _subset_sums(final)
        
        if len(final) > len(best):
            best = final[:]
    
    # Final validation
    if not _check_validity_fast(best):
        best = [31, 32, 34]
    
    # Ensure all are Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
