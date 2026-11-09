"""Optimized subset-sum divisibility search with greedy + local search."""
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

def _check_validity(nums):
    """Check divisibility-incomparable subset sum constraint."""
    n = len(nums)
    if n == 0:
        return True
    if n > 20:
        return False  # Too many sums to check efficiently
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    sums_list = sorted(sums)
    sums_set = set(sums_list)
    max_sum = sums_list[-1]
    for s in sums_list:
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return False
    return True

def _check_validity_fast(nums):
    """Faster validity check with early termination."""
    n = len(nums)
    if n == 0:
        return True
    if n > 20:
        return False
    sums = _subset_sums(nums)
    if len(sums) != (1 << n) - 1:
        return False
    sums_list = sorted(sums)
    sums_set = set(sums_list)
    max_sum = sums_list[-1]
    min_sum = sums_list[0]
    # Quick check: if max < 2*min, no divisibility possible
    if max_sum < 2 * min_sum:
        return True
    for s in sums_list:
        # Only need to check multiples up to max_sum
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return False
    return True

def _try_add(current, candidate, deadline):
    """Try adding candidate to current set, return True if valid."""
    if time.time() > deadline:
        return False
    trial = current + [candidate]
    return _check_validity_fast(trial)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Start with a known valid set
    best = [128, 131, 134, 137, 140, 143, 146]
    
    # Verify initial best
    if not _check_validity(best):
        best = [100, 103, 106]
    
    all_candidates = list(range(1, 257))
    
    # Strategy 1: Greedy construction from high range
    # Try multiple starting points
    for base in range(200, 50, -5):
        if time.time() > deadline - 10:
            break
        
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            rng = random.Random(trial * 7919 + base)
            
            # Create candidate pool biased toward larger numbers
            pool = list(range(base, 257))
            pool += list(range(base - 30, base))
            rng.shuffle(pool)
            
            current = []
            for num in pool:
                if len(current) >= 12:
                    break
                if _try_add(current, num, deadline):
                    current.append(num)
            
            if len(current) > len(best):
                best = current[:]
    
    # Strategy 2: Extend the best set
    if time.time() < deadline - 15:
        extended = best[:]
        # Try adding candidates in descending order
        for num in sorted(all_candidates, reverse=True):
            if num in extended:
                continue
            if _try_add(extended, num, deadline):
                extended.append(num)
                if len(extended) > len(best):
                    best = extended[:]
    
    # Strategy 3: Local search - replace elements to enable growth
    if time.time() < deadline - 10:
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(iteration * 31337 + int(start_time * 1000))
            current = best[:]
            improved = False
            
            # Try replacing one element
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                
                old_val = current[i]
                # Try several replacements
                replacements = rng.sample([x for x in all_candidates if x != old_val], 30)
                for repl in replacements:
                    trial = current[:]
                    trial[i] = repl
                    if _check_validity_fast(trial):
                        # Check if we can now add more elements
                        extended_trial = trial[:]
                        for num in sorted(all_candidates, reverse=True):
                            if num in extended_trial:
                                continue
                            if _try_add(extended_trial, num, deadline):
                                extended_trial.append(num)
                                if len(extended_trial) > len(best):
                                    best = extended_trial[:]
                                    current = extended_trial[:]
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
            
            if not improved and iteration > 50:
                # Try adding a random element
                for num in rng.sample(all_candidates, 20):
                    if num in current:
                        continue
                    if _try_add(current, num, deadline):
                        current.append(num)
                        if len(current) > len(best):
                            best = current[:]
                        break
    
    # Strategy 4: Try a completely different range
    if time.time() < deadline - 10:
        for start_range in [150, 180, 210, 220]:
            if time.time() > deadline - 5:
                break
            
            for trial in range(30):
                if time.time() > deadline - 5:
                    break
                
                rng = random.Random(trial * 104729 + start_range)
                pool = list(range(start_range, 257))
                if start_range > 20:
                    pool += list(range(start_range - 20, start_range))
                rng.shuffle(pool)
                
                current = []
                for num in pool:
                    if len(current) >= 12:
                        break
                    if _try_add(current, num, deadline):
                        current.append(num)
                
                if len(current) > len(best):
                    best = current[:]
    
    # Final validation
    if not _check_validity(best):
        # Fallback to a definitely valid small set
        best = [100, 101, 103]
        if not _check_validity(best):
            best = [100, 101]
    
    return list(best)
# EVOLVE-BLOCK-END
