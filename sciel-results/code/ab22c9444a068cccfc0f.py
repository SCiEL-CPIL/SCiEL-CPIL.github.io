"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _check_divisibility_conflict(new_sums, existing_sums):
    """Check if any pair of sums (one from new, one from existing, or two from new)
    have a divisibility relationship. Returns True if conflict found."""
    # Check new sums against existing sums
    for ns in new_sums:
        for es in existing_sums:
            if ns != es:
                if ns % es == 0 or es % ns == 0:
                    return True
    # Check new sums among themselves
    n = len(new_sums)
    for i in range(n):
        for j in range(i+1, n):
            a, b = new_sums[i], new_sums[j]
            if a != b:
                if a % b == 0 or b % a == 0:
                    return True
    return False

def _try_build_set(order, max_time):
    """Try to build a valid set by adding elements in the given order."""
    selected = []
    all_sums = set()  # all nonempty subset sums
    
    start_time = time.time()
    
    for candidate in order:
        if time.time() - start_time > max_time:
            break
        
        # Compute new subset sums when adding candidate
        # New sums = {candidate} ∪ {s + candidate for s in all_sums}
        new_sums = [candidate] + [s + candidate for s in all_sums]
        
        # Check for divisibility conflicts
        if _check_divisibility_conflict(new_sums, all_sums):
            continue
        
        # No conflict, add candidate
        selected.append(candidate)
        all_sums.update(new_sums)
    
    return selected

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    best_size = 0
    best_set = [31, 32, 34]
    
    # Strategy 1: Larger-first greedy (larger numbers are less likely to divide others)
    candidates_desc = list(range(256, 0, -1))
    result = _try_build_set(candidates_desc, 2.0)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result
    
    # Strategy 2: Random restarts with various orderings
    random.seed(42)
    
    for trial in range(200):
        if time.time() - start_time > deadline - 5.0:
            break
        
        # Create a random ordering with bias toward larger numbers
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Bias: sort by a mix of value and random noise
        # Larger numbers tend to be better (less likely to divide)
        biased_order = sorted(candidates, key=lambda x: -x + random.random() * 100)
        
        result = _try_build_set(biased_order, 0.5)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Strategy 3: Try starting with large primes (they're hard to divide)
    if time.time() - start_time < deadline - 5.0:
        primes = [x for x in range(2, 257) if all(x % i != 0 for i in range(2, int(x**0.5)+1))]
        # Start with large primes, then fill in
        primes_desc = sorted(primes, reverse=True)
        others = [x for x in range(1, 257) if x not in primes]
        others_desc = sorted(others, reverse=True)
        
        # Try different mixes
        for p_count in [10, 20, 30, 40]:
            if time.time() - start_time > deadline - 5.0:
                break
            order = primes_desc[:p_count] + others_desc
            result = _try_build_set(order, 1.0)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result
    
    # Strategy 4: Backtracking with forward checking for better packing
    if time.time() - start_time < deadline - 10.0:
        # Use a smarter approach: try to add elements that create fewest conflicts
        remaining = set(range(1, 257))
        selected = []
        all_sums = set()
        
        while remaining and time.time() - start_time < deadline - 5.0:
            best_candidate = None
            best_conflict_count = float('inf')
            
            # Sample a subset of candidates to check (for speed)
            candidates_to_try = list(remaining)
            if len(candidates_to_try) > 50:
                random.shuffle(candidates_to_try)
                candidates_to_try = candidates_to_try[:50]
            
            for candidate in candidates_to_try:
                new_sums = [candidate] + [s + candidate for s in all_sums]
                
                # Count conflicts
                conflict_count = 0
                for ns in new_sums:
                    for es in all_sums:
                        if ns != es and (ns % es == 0 or es % ns == 0):
                            conflict_count += 1
                            if conflict_count > best_conflict_count:
                                break
                    if conflict_count > best_conflict_count:
                        break
                
                if conflict_count == 0:
                    best_candidate = candidate
                    best_conflict_count = 0
                    break
                elif conflict_count < best_conflict_count:
                    best_conflict_count = conflict_count
                    best_candidate = candidate
            
            if best_candidate is not None and best_conflict_count == 0:
                new_sums = [best_candidate] + [s + best_candidate for s in all_sums]
                selected.append(best_candidate)
                all_sums.update(new_sums)
                remaining.discard(best_candidate)
            else:
                break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
    
    # Ensure we return a valid result
    if len(best_set) < 3:
        best_set = [31, 32, 34]
    
    return list(best_set)
# EVOLVE-BLOCK-END
