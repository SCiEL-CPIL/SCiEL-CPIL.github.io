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

def _constructive_interval_search(start_time, deadline, best_size, best_set):
    """Constructive approach: select integers from narrow intervals where
    subset sums of different cardinalities fall into disjoint ranges, reducing
    cross-cardinality divisibility conflicts."""
    # Try intervals [2^k, 2^k + 2^(k-1)) for various k
    # For k=7: [128, 192), for k=8: [256, 384) - too large, max is 256
    # So k=7: [128, 191] (64 values), k=6: [64, 95] (32 values)
    # Also try [100, 150], [120, 180], etc.
    
    intervals = [
        list(range(128, 192)),
        list(range(100, 160)),
        list(range(120, 190)),
        list(range(150, 257)),
        list(range(80, 140)),
        list(range(128, 257)),
        list(range(100, 257)),
    ]
    
    for interval in intervals:
        if time.time() - start_time > deadline - 5.0:
            break
        
        # Shuffle and try greedy
        random.shuffle(interval)
        result = _try_build_set(interval, 1.5)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
        
        # Also try sorted descending
        result = _try_build_set(sorted(interval, reverse=True), 1.0)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    return best_size, best_set

def _backtrack_improve(current_set, start_time, deadline):
    """Try to improve a set by backtracking: remove elements and try alternatives."""
    if not current_set:
        return current_set
    
    selected = list(current_set)
    all_sums = set()
    
    # Rebuild sums
    for e in selected:
        new_sums = [e] + [s + e for s in all_sums]
        all_sums.update(new_sums)
    
    remaining = set(range(1, 257)) - set(selected)
    improved = False
    
    while remaining and time.time() - start_time < deadline - 5.0:
        best_candidate = None
        best_conflict_count = float('inf')
        
        candidates_to_try = list(remaining)
        if len(candidates_to_try) > 60:
            random.shuffle(candidates_to_try)
            candidates_to_try = candidates_to_try[:60]
        
        for candidate in candidates_to_try:
            new_sums = [candidate] + [s + candidate for s in all_sums]
            
            # Check for exact validity (no conflicts)
            conflict = False
            for ns in new_sums:
                for es in all_sums:
                    if ns != es and (ns % es == 0 or es % ns == 0):
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                # Check new sums among themselves
                for i in range(len(new_sums)):
                    for j in range(i+1, len(new_sums)):
                        a, b = new_sums[i], new_sums[j]
                        if a != b and (a % b == 0 or b % a == 0):
                            conflict = True
                            break
                    if conflict:
                        break
            
            if not conflict:
                best_candidate = candidate
                break
        
        if best_candidate is not None:
            new_sums = [best_candidate] + [s + best_candidate for s in all_sums]
            selected.append(best_candidate)
            all_sums.update(new_sums)
            remaining.discard(best_candidate)
            improved = True
        else:
            break
    
    if improved:
        return selected
    return current_set

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    best_size = 0
    best_set = [31, 32, 34]
    
    # Strategy 1: Larger-first greedy
    candidates_desc = list(range(256, 0, -1))
    result = _try_build_set(candidates_desc, 2.0)
    if len(result) > best_size:
        best_size = len(result)
        best_set = result
    
    # Strategy 2: Constructive interval search (new)
    best_size, best_set = _constructive_interval_search(start_time, deadline, best_size, best_set)
    
    # Strategy 3: Random restarts with various orderings
    random.seed(42)
    
    for trial in range(300):
        if time.time() - start_time > deadline - 5.0:
            break
        
        # Create a random ordering with bias toward larger numbers
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Bias: sort by a mix of value and random noise
        biased_order = sorted(candidates, key=lambda x: -x + random.random() * 100)
        
        result = _try_build_set(biased_order, 0.3)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Strategy 4: Try starting with large primes
    if time.time() - start_time < deadline - 5.0:
        primes = [x for x in range(2, 257) if all(x % i != 0 for i in range(2, int(x**0.5)+1))]
        primes_desc = sorted(primes, reverse=True)
        others = [x for x in range(1, 257) if x not in primes]
        others_desc = sorted(others, reverse=True)
        
        for p_count in [10, 20, 30, 40]:
            if time.time() - start_time > deadline - 5.0:
                break
            order = primes_desc[:p_count] + others_desc
            result = _try_build_set(order, 1.0)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result
    
    # Strategy 5: Backtracking improvement on best set found
    if time.time() - start_time < deadline - 10.0:
        improved = _backtrack_improve(best_set, start_time, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Strategy 6: More random restarts focused on large numbers
    if time.time() - start_time < deadline - 5.0:
        for trial in range(200):
            if time.time() - start_time > deadline - 3.0:
                break
            
            # Focus on numbers > 100
            large_candidates = list(range(100, 257))
            random.shuffle(large_candidates)
            result = _try_build_set(large_candidates, 0.4)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result
    
    # Strategy 7: Final polish - try to add one more element to best set
    if time.time() - start_time < deadline - 3.0:
        improved = _backtrack_improve(best_set, start_time, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Ensure we return a valid result
    if len(best_set) < 3:
        best_set = [31, 32, 34]
    
    return list(best_set)
# EVOLVE-BLOCK-END
