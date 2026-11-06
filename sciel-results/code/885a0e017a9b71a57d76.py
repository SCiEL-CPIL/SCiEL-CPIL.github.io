import time
import random
import bisect
import numpy as np

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    def is_valid_addition(s_sorted, candidate):
        """Check if adding candidate to s_sorted (sorted list) keeps validity.
        Returns True if valid, False otherwise."""
        n = len(s_sorted)
        pos = bisect.bisect_left(s_sorted, candidate)
        
        # Check 1: candidate as 'a' (smallest element in triple)
        # For all b, c in s_sorted with b > candidate, c >= b:
        #   candidate must NOT divide (b + c)
        # i.e., for all i >= pos, j >= i: s_sorted[j] % candidate != (-s_sorted[i]) % candidate
        for i in range(pos, n):
            b = s_sorted[i]
            target = (-b) % candidate
            # Check if any s_sorted[k] for k >= i has s_sorted[k] % candidate == target
            for k in range(i, n):
                if s_sorted[k] % candidate == target:
                    return False
        
        # Check 2: candidate as 'b' (middle element in triple)
        # For all a < candidate in s_sorted, c >= candidate in {candidate} union s_sorted[pos:]:
        #   a must NOT divide (candidate + c)
        for i in range(pos):
            a = s_sorted[i]
            # c can be candidate itself
            if (candidate + candidate) % a == 0:
                return False
            # c can be any s_sorted[k] for k >= pos
            target = (-candidate) % a
            for k in range(pos, n):
                if s_sorted[k] % a == target:
                    return False
        
        # Check 3: candidate as 'c' (largest element in triple)
        # For all a < b <= candidate in s_sorted:
        #   a must NOT divide (b + candidate)
        for i in range(n):
            a = s_sorted[i]
            for j in range(i + 1, n):
                b = s_sorted[j]
                if b > candidate:
                    break
                if (b + candidate) % a == 0:
                    return False
        
        return True
    
    def try_add(s_sorted, candidate):
        """Try to add candidate to s_sorted. Returns (success, new_sorted)."""
        if is_valid_addition(s_sorted, candidate):
            pos = bisect.bisect_left(s_sorted, candidate)
            new_s = s_sorted[:]
            new_s.insert(pos, candidate)
            return True, new_s
        return False, s_sorted
    
    # Strategy: Start with upper range, repair, then greedily add smaller elements
    best_set = []
    best_len = 0
    
    # Try different starting points
    for min_start in [128, 140, 150, 160, 171, 180, 192, 200, 210, 220]:
        if time.time() - start_time > deadline:
            break
        
        # Start with all elements from min_start to N
        s_sorted = list(range(min_start, N + 1))
        s_set = set(s_sorted)
        
        # Repair: iteratively find and remove conflicting elements
        changed = True
        while changed:
            changed = False
            for i in range(len(s_sorted)):
                a = s_sorted[i]
                found = False
                for j in range(i + 1, len(s_sorted)):
                    b = s_sorted[j]
                    target = (-b) % a
                    for k in range(j, len(s_sorted)):
                        if s_sorted[k] % a == target:
                            # Remove a (the smallest, most constraining)
                            s_sorted.pop(i)
                            s_set.discard(a)
                            changed = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        # Now greedily add elements from 1 to min_start-1 (ascending)
        for candidate in range(1, min_start):
            if candidate in s_set:
                continue
            ok, new_s = try_add(s_sorted, candidate)
            if ok:
                s_sorted = new_s
                s_set.add(candidate)
        
        # Also try adding from min_start-1 down to 1 (descending) in a fresh copy
        s_sorted2 = list(range(min_start, N + 1))
        s_set2 = set(s_sorted2)
        # Repair s_sorted2
        changed = True
        while changed:
            changed = False
            for i in range(len(s_sorted2)):
                a = s_sorted2[i]
                found = False
                for j in range(i + 1, len(s_sorted2)):
                    b = s_sorted2[j]
                    target = (-b) % a
                    for k in range(j, len(s_sorted2)):
                        if s_sorted2[k] % a == target:
                            s_sorted2.pop(i)
                            s_set2.discard(a)
                            changed = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        for candidate in range(min_start - 1, 0, -1):
            if candidate in s_set2:
                continue
            ok, new_s = try_add(s_sorted2, candidate)
            if ok:
                s_sorted2 = new_s
                s_set2.add(candidate)
        
        # Keep the better of the two
        if len(s_sorted) >= len(s_sorted2):
            if len(s_sorted) > best_len:
                best_set = s_sorted[:]
                best_len = len(s_sorted)
        else:
            if len(s_sorted2) > best_len:
                best_set = s_sorted2[:]
                best_len = len(s_sorted2)
    
    # Local search: systematic swap-based improvement
    # For each element in the set, try removing it and adding a different one
    s_sorted = best_set[:]
    s_set = set(s_sorted)
    
    max_iters = 500
    for iteration in range(max_iters):
        if time.time() - start_time > deadline:
            break
        
        improved = False
        n_curr = len(s_sorted)
        
        # Try each element as a removal candidate
        for rem_idx in range(n_curr):
            if time.time() - start_time > deadline:
                break
            removed = s_sorted[rem_idx]
            s_without = s_sorted[:rem_idx] + s_sorted[rem_idx+1:]
            s_without_set = set(s_without)
            
            # Try adding each missing element
            for candidate in range(1, N + 1):
                if candidate in s_without_set:
                    continue
                ok, new_s = try_add(s_without, candidate)
                if ok:
                    if len(new_s) > len(s_sorted):
                        s_sorted = new_s
                        s_set = set(new_s)
                        improved = True
                        break
                    elif len(new_s) == len(s_sorted) and candidate < removed:
                        # Same size but smaller element: might allow further additions later
                        # Only accept if it opens up more possibilities (heuristic: accept)
                        s_sorted = new_s
                        s_set = set(new_s)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try random swaps
            for _ in range(50):
                if time.time() - start_time > deadline:
                    break
                rem_idx = random.randint(0, len(s_sorted) - 1)
                removed = s_sorted[rem_idx]
                s_without = s_sorted[:rem_idx] + s_sorted[rem_idx+1:]
                s_without_set = set(s_without)
                
                # Pick a random missing element
                missing = [x for x in range(1, N + 1) if x not in s_without_set]
                if not missing:
                    break
                candidate = random.choice(missing)
                ok, new_s = try_add(s_without, candidate)
                if ok and len(new_s) >= len(s_sorted):
                    s_sorted = new_s
                    s_set = set(new_s)
                    improved = True
                    break
            
            if not improved:
                # No improvement found, try a different random perturbation
                if len(s_sorted) > 2:
                    # Remove 2 random elements, try to add 2-3 others
                    idxs = random.sample(range(len(s_sorted)), 2)
                    idxs.sort(reverse=True)
                    for idx in idxs:
                        s_set.discard(s_sorted[idx])
                        s_sorted.pop(idx)
                    
                    # Try to add back
                    s_set = set(s_sorted)
                    for candidate in range(1, N + 1):
                        if candidate in s_set:
                            continue
                        ok, new_s = try_add(s_sorted, candidate)
                        if ok:
                            s_sorted = new_s
                            s_set.add(candidate)
                    
                    if len(s_sorted) > len(best_set):
                        best_set = s_sorted[:]
                        best_len = len(s_sorted)
                break
    
    # Final: ensure best_set is the larger of s_sorted and best_set
    if len(s_sorted) > len(best_set):
        best_set = s_sorted[:]
    
    return best_set
