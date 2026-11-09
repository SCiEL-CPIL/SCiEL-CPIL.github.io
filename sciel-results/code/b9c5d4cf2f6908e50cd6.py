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

def _check_validity(nums):
    """Check divisibility-incomparable subset sum constraint."""
    n = len(nums)
    if n == 0:
        return True
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

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best = [31, 32, 34]
    
    # Backtracking/greedy search
    # Try to build valid sets of increasing size
    
    # Candidate pool: focus on larger numbers to reduce divisibility risk
    all_candidates = list(range(1, 257))
    
    # Strategy 1: Greedy with multiple random orderings on different ranges
    ranges_to_try = [
        list(range(100, 257)),
        list(range(80, 257)),
        list(range(60, 257)),
        list(range(40, 257)),
        all_candidates,
    ]
    
    for cand_range in ranges_to_try:
        if time.time() > deadline - 10:
            break
        
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            
            rng = random.Random(trial * 7919 + len(cand_range))
            order = cand_range[:]
            rng.shuffle(order)
            
            current = []
            for num in order:
                trial_set = current + [num]
                if len(trial_set) <= 12:
                    if _check_validity(trial_set):
                        current = trial_set
                else:
                    break
            
            if len(current) > len(best):
                best = current[:]
        
        # Try to extend best further with remaining candidates
        if time.time() > deadline - 10:
            break
        extended = best[:]
        for num in sorted(all_candidates, reverse=True):
            if num in extended:
                continue
            trial_set = extended + [num]
            if len(trial_set) <= 14:
                if _check_validity(trial_set):
                    extended = trial_set
        if len(extended) > len(best):
            best = extended[:]
    
    # Final local search: try swapping elements
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            new_best = best[:]
            improved = False
            for i in range(len(new_best)):
                for replacement in random.sample(all_candidates, 30):
                    if replacement in new_best:
                        continue
                    trial = new_best[:]
                    trial[i] = replacement
                    if _check_validity(trial):
                        new_best = trial
                        improved = True
                        break
                if improved:
                    break
            if len(new_best) > len(best):
                best = new_best[:]
    
    # Final validation
    if not _check_validity(best):
        best = [31, 32, 34]
    
    return list(best)
# EVOLVE-BLOCK-END
