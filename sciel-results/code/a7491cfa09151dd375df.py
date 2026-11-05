import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 12
    # Fallback: powers of 2 (valid, largest = 2048)
    best = [1 << i for i in range(12)]
    best_max = max(best)
    
    random.seed(42)
    
    # Greedy construction with backtracking-like search
    # Key insight: for distinct subset sums, we need a "Sidon-like" property for subset sums.
    # A sufficient condition for distinct subset sums is that the sequence is "complete" 
    # in the sense that each element is greater than the sum of all previous elements 
    # (superincreasing), but that gives powers of 2.
    # 
    # A better approach: use the fact that if we sort the numbers a1 < a2 < ... < an,
    # then a necessary and sufficient condition for all subset sums to be distinct is that
    # no element equals the difference of two subset sums of the remaining elements.
    # 
    # Greedy approach: pick the smallest valid number at each step.
    # For the k-th number, it must not create a collision with existing subset sums.
    # 
    # Let's implement a smarter greedy with some randomization to explore different paths.
    
    def is_valid_partial(arr):
        """Check if all subset sums of arr are distinct."""
        if not arr:
            return True
        sums = set([0])
        for x in arr:
            new_sums = set(s + x for s in sums)
            if sums & new_sums:
                return False
            sums = sums | new_sums
        return True
    
    def greedy_construct(max_steps=100000):
        """Greedy construction with randomization."""
        arr = []
        sums = set([0])
        
        for k in range(n):
            # Try candidates starting from 1
            # But to be smarter, we can start from a lower bound
            # The new element x must satisfy: (sums + x) is disjoint from sums
            # i.e., x is not in {a - b | a, b in sums, a != b}
            
            # For efficiency, let's just try incrementally
            # But we can skip some values if we know they'll collide
            
            candidate = 1
            found = False
            
            while not found:
                if time.time() > deadline:
                    return None
                
                # Check distinctness
                if candidate in arr:
                    candidate += 1
                    continue
                
                # Check subset sum collision
                # (sums + candidate) must be disjoint from sums
                # Equivalent: candidate not in difference set of sums
                
                # Quick check: if candidate > max(sums), it's automatically valid
                # because all new sums will be > max(sums) >= min(sums)
                if candidate > max(sums):
                    found = True
                else:
                    # Check intersection
                    # shifted = {s + candidate for s in sums}
                    # if shifted & sums: invalid
                    # Optimization: only need to check if any s + candidate is in sums
                    # i.e., if candidate is in {s1 - s2 for s1, s2 in sums}
                    # But that's expensive. Let's just check directly for small sums.
                    
                    # For small k, sums is small, so this is fast
                    collision = False
                    for s in sums:
                        if (s + candidate) in sums:
                            collision = True
                            break
                    
                    if not collision:
                        found = True
                    else:
                        candidate += 1
                
                # Safety: don't loop forever
                if candidate > 10000000:
                    return None
            
            arr.append(candidate)
            sums = sums | {s + candidate for s in sums}
        
        return arr
    
    # First, try pure greedy (deterministic)
    result = greedy_construct()
    if result is not None and len(result) == n:
        m = max(result)
        if m < best_max:
            best = result
            best_max = m
    
    # Now try randomized greedy: at each step, sometimes skip the smallest valid
    # and try a slightly larger one, hoping to enable smaller later choices.
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        arr = []
        sums = set([0])
        valid = True
        
        for k in range(n):
            if time.time() > deadline:
                valid = False
                break
            
            # Find the smallest valid candidate
            candidate = 1
            min_valid = None
            
            # For randomization, we'll find several valid candidates and pick one
            # But that's expensive. Instead, let's just try the smallest, but with
            # some probability, try a slightly larger one.
            
            # Find smallest valid
            while True:
                if time.time() > deadline:
                    valid = False
                    break
                
                if candidate in arr:
                    candidate += 1
                    continue
                
                if candidate > max(sums):
                    min_valid = candidate
                    break
                else:
                    collision = False
                    for s in sums:
                        if (s + candidate) in sums:
                            collision = True
                            break
                    
                    if not collision:
                        min_valid = candidate
                        break
                    else:
                        candidate += 1
            
            if not valid:
                break
            
            if min_valid is None:
                valid = False
                break
            
            # With some probability, try a larger value to explore different paths
            # This is a heuristic: sometimes picking a larger element early can
            # allow smaller elements later, reducing the overall max.
            if random.random() < 0.2 and k < n - 1:
                # Try a few larger values
                for offset in range(1, 5):
                    test_val = min_valid + offset
                    if test_val in arr:
                        continue
                    
                    if test_val > max(sums):
                        arr.append(test_val)
                        sums = sums | {s + test_val for s in sums}
                        break
                    else:
                        collision = False
                        for s in sums:
                            if (s + test_val) in sums:
                                collision = True
                                break
                        
                        if not collision:
                            arr.append(test_val)
                            sums = sums | {s + test_val for s in sums}
                            break
                        else:
                            continue
                else:
                    # No larger value worked, use the smallest
                    arr.append(min_valid)
                    sums = sums | {s + min_valid for s in sums}
            else:
                arr.append(min_valid)
                sums = sums | {s + min_valid for s in sums}
        
        if valid and len(arr) == n:
            m = max(arr)
            if m < best_max:
                best = arr
                best_max = m
    
    # Final local search: try to reduce the largest element
    # Sort best, try to replace the largest with a smaller valid value
    best_sorted = sorted(best)
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Try to reduce the last element
        last_idx = len(best_sorted) - 1
        old_val = best_sorted[last_idx]
        
        # Try values from old_val - 1 down to 1
        for new_val in range(old_val - 1, 0, -1):
            if time.time() > deadline:
                break
            
            if new_val in best_sorted:
                continue
            
            # Check if replacing last element with new_val keeps all subset sums distinct
            # Remove last element, compute subset sums of remaining 11
            remaining = best_sorted[:last_idx]
            sums_remaining = set([0])
            for x in remaining:
                sums_remaining = sums_remaining | {s + x for s in sums_remaining}
            
            # Check if new_val creates collision
            collision = False
            for s in sums_remaining:
                if (s + new_val) in sums_remaining:
                    collision = True
                    break
            
            if not collision:
                # Valid! Update best
                best_sorted[last_idx] = new_val
                best_sorted = sorted(best_sorted)
                best_max = max(best_sorted)
                break
    
    return best_sorted
