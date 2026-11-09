"""Optimized subset-sum divisibility search with fast validity check."""
# EVOLVE-BLOCK-START
import time
import random

def _check_validity(nums):
    """Check divisibility-incomparable subset sum constraint. Returns True if valid."""
    n = len(nums)
    if n == 0:
        return True
    if n > 15:
        return False  # Too many sums to check quickly
    
    # Generate all subset sums
    sums = set()
    for x in nums:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        sums |= new_sums
    
    # Check all subset sums are distinct
    if len(sums) != (1 << n) - 1:
        return False
    
    # Check no two sums divide each other
    sums_list = sorted(sums)
    sums_set = set(sums_list)
    max_sum = sums_list[-1]
    for s in sums_list:
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return False
    return True

def _check_validity_incremental(current_sums, new_elem, n_before):
    """Check if adding new_elem to a set with existing sums maintains validity.
    current_sums: set of existing subset sums
    new_elem: the new element to add
    n_before: number of elements before adding
    Returns (new_sums, is_valid) or (None, False) if invalid.
    """
    # Generate new sums by adding new_elem to all existing sums
    new_sums = set(current_sums)
    for s in current_sums:
        new_sums.add(s + new_elem)
    new_sums.add(new_elem)
    
    # Check distinctness
    n_after = n_before + 1
    if len(new_sums) != (1 << n_after) - 1:
        return None, False
    
    # Check divisibility: for each sum, check multiples
    sums_list = sorted(new_sums)
    sums_set = new_sums
    max_sum = sums_list[-1]
    for s in sums_list:
        for k in range(2, max_sum // s + 1):
            if k * s in sums_set:
                return None, False
    
    return new_sums, True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s limit
    
    # Start with a known valid set
    best = [128, 129, 131, 134, 140, 152, 176]
    if not _check_validity(best):
        best = [31, 32, 34]
    
    all_candidates = list(range(1, 257))
    
    # Greedy construction with incremental validity check
    # Try multiple starting points and orderings
    best_size = len(best)
    
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(trial * 7919 + 13)
        
        # Choose a random range for candidates
        lo = rng.randint(50, 200)
        hi = rng.randint(lo + 10, 256)
        pool = list(range(lo, hi + 1))
        rng.shuffle(pool)
        
        current = []
        current_sums = set()
        
        for num in pool:
            if len(current) >= 14:
                break
            new_sums, valid = _check_validity_incremental(current_sums, num, len(current))
            if valid:
                current.append(num)
                current_sums = new_sums
        
        if len(current) > best_size:
            best_size = len(current)
            best = current[:]
    
    # Try to extend the best set further
    if time.time() < deadline - 15:
        # Compute current sums for best
        best_sums = set()
        for x in best:
            new_sums = set()
            for s in best_sums:
                new_sums.add(s + x)
            new_sums.add(x)
            best_sums |= new_sums
        
        # Try adding elements from high to low
        extended = best[:]
        extended_sums = best_sums
        for num in sorted(all_candidates, reverse=True):
            if num in extended:
                continue
            if len(extended) >= 15:
                break
            new_sums, valid = _check_validity_incremental(extended_sums, num, len(extended))
            if valid:
                extended.append(num)
                extended_sums = new_sums
        
        if len(extended) > best_size:
            best_size = len(extended)
            best = extended[:]
    
    # Local search: try swapping elements to enable growth
    if time.time() < deadline - 10:
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            
            new_best = best[:]
            improved = False
            
            for i in range(len(new_best)):
                if time.time() > deadline - 10:
                    break
                # Try replacing element i
                for replacement in rng.sample(all_candidates, 30):
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
            
            # After a swap, try to extend
            if improved:
                # Recompute sums
                nb_sums = set()
                for x in new_best:
                    ns = set()
                    for s in nb_sums:
                        ns.add(s + x)
                    ns.add(x)
                    nb_sums |= ns
                
                for num in sorted(all_candidates, reverse=True):
                    if num in new_best:
                        continue
                    if len(new_best) >= 15:
                        break
                    new_sums, valid = _check_validity_incremental(nb_sums, num, len(new_best))
                    if valid:
                        new_best.append(num)
                        nb_sums = new_sums
            
            if len(new_best) > best_size:
                best_size = len(new_best)
                best = new_best[:]
    
    # Final validation
    if not _check_validity(best):
        best = [31, 32, 34]
    
    return list(best)
# EVOLVE-BLOCK-END
