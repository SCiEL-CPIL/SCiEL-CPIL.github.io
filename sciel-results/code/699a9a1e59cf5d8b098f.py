"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _is_valid(s):
    """Check if set s (list of ints) satisfies: for all a < b <= c in s, a does not divide b+c."""
    s_set = set(s)
    for i, a in enumerate(s):
        if a == 0:
            continue
        for b in s[i+1:]:
            for c in s[i+1:]:
                if c >= b and (b + c) % a == 0:
                    return False
    return True

def _can_add(s, candidate):
    """Check if adding candidate to sorted list s maintains validity.
    Only checks triples involving candidate."""
    # Find where candidate would be inserted
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi) // 2
        if s[mid] < candidate:
            lo = mid + 1
        else:
            hi = mid
    idx = lo
    
    n = len(s)
    
    # Case 1: candidate as a (smallest)
    # For all b > candidate in s, c >= b in s: candidate must not divide b+c
    for j in range(idx, n):
        b = s[j]
        if b <= candidate:
            continue
        target = (-b) % candidate
        for k in range(j, n):
            c = s[k]
            if c >= b and c % candidate == target:
                return False
    # Also need to check if candidate itself can be b or c with itself? 
    # No, a < b <= c, so a must be strictly less. If candidate is a, b > candidate.
    
    # Case 2: candidate as b (middle)
    # For all a < candidate in s, c >= candidate in s (including candidate itself if c=candidate? No, c >= b=candidate)
    # Wait, the condition is a < b <= c.
    # If candidate is b, then a < candidate and c >= candidate.
    # c can be candidate itself? Yes, if c = candidate, then b = candidate, so b <= c holds.
    # But c must be in s. Since we are adding candidate, c can be candidate.
    # So we check a in s with a < candidate, and c in {candidate} U {x in s | x >= candidate}
    for i in range(idx):
        a = s[i]
        if a >= candidate:
            break
        # c ranges over elements in s that are >= candidate, plus candidate itself
        # Let's iterate c from s[idx:] and also candidate
        # Actually, since s is sorted, s[idx:] are all >= candidate (if idx is insertion point for < candidate)
        # Wait, if s[idx] == candidate, then candidate is already in s? No, we are checking if we CAN add it.
        # If candidate is already in s, we shouldn't be calling this.
        # So s[idx] > candidate or idx == n.
        # So s[idx:] are all > candidate.
        # c must be >= candidate. So c can be candidate, or any x in s[idx:].
        
        # Check c = candidate
        if (candidate + candidate) % a == 0:
            return False
        
        # Check c in s[idx:]
        target = (-candidate) % a
        for k in range(idx, n):
            c = s[k]
            # c is > candidate, so c >= candidate holds
            if c % a == target:
                return False
    
    # Case 3: candidate as c (largest)
    # For all a < b <= candidate in s: a must not divide b + candidate
    # b must be < candidate (since b <= c=candidate and a < b, so b < candidate unless b=candidate)
    # Wait, if b = candidate, then a < candidate.
    # So b ranges over s elements that are < candidate, and also candidate itself?
    # If b = candidate, then c = candidate. This is covered in Case 2?
    # In Case 2, candidate was b. Here candidate is c.
    # If b = candidate and c = candidate, then a < candidate.
    # In Case 2, we checked a < candidate, c = candidate. That covers (a, candidate, candidate).
    # So for Case 3, we only need to consider b < candidate.
    # a < b < candidate.
    # a and b are in s.
    for i in range(idx):
        a = s[i]
        if a >= candidate:
            break
        for j in range(i + 1, idx):
            b = s[j]
            # b < candidate
            if (b + candidate) % a == 0:
                return False
    
    return True

def _greedy_construct(start_set, deadline=85.0):
    """Greedily expand start_set to a maximal valid set."""
    s = sorted(start_set)
    # Repair: remove conflicts
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        # Find element with most conflicts
        # Approximate: for each element, count how many triples it violates
        # This is expensive. Instead, just try removing each element and see if it helps?
        # Or, simply remove the smallest element if it causes conflicts?
        # Let's do a simple repair: if not valid, remove the element that is part of the most violations.
        # To avoid O(n^3) per iteration, we'll just remove the smallest element that is involved in a violation.
        
        # Check validity
        s_set = set(s)
        violation_found = False
        for i, a in enumerate(s):
            if a == 0:
                continue
            for j in range(i + 1, len(s)):
                b = s[j]
                target = (-b) % a
                for k in range(j, len(s)):
                    c = s[k]
                    if c % a == target:
                        # Violation found involving a, b, c
                        # Remove the smallest one? Or the one with smallest value?
                        # Removing smallest a might be best
                        s.remove(a)
                        changed = True
                        violation_found = True
                        break
                if violation_found:
                    break
            if violation_found:
                break
        
        if not violation_found:
            break
    
    # Now greedy expand
    for candidate in range(1, N + 1):
        if time.time() > deadline:
            break
        if candidate not in s:
            if _can_add(s, candidate):
                # Insert in sorted order
                lo, hi = 0, len(s)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if s[mid] < candidate:
                        lo = mid + 1
                    else:
                        hi = mid
                s.insert(lo, candidate)
    
    return s

def _local_search_swap(current_set, max_iterations=100, deadline=85.0):
    """Local search: try swapping one element out and another in to increase size or maintain size with better structure."""
    s = list(current_set)
    s.sort()
    best = list(s)
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        if len(s) <= 1:
            break
        
        # Try removing a random element
        idx = random.randint(0, len(s) - 1)
        removed = s[idx]
        s.pop(idx)
        
        # Try to add back other elements to restore or increase size
        found_better = False
        for candidate in range(1, N + 1):
            if candidate in s:
                continue
            if _can_add(s, candidate):
                # Insert
                lo, hi = 0, len(s)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if s[mid] < candidate:
                        lo = mid + 1
                    else:
                        hi = mid
                s.insert(lo, candidate)
                if len(s) > len(best):
                    best = list(s)
                found_better = True
                break
        
        if not found_better:
            # Restore removed element if we couldn't find a replacement
            s.append(removed)
            s.sort()
    
    return best

def construct():
    start_time = time.time()
    deadline = 85.0  # 90s limit, leave 5s margin
    
    best_set = list(range(192, 257))
    
    random.seed(42)
    
    # Strategy 1: Upper half
    candidate1 = list(range(128, 257))
    repaired1 = _greedy_construct(candidate1, deadline)
    if len(repaired1) > len(best_set):
        best_set = repaired1
    
    if time.time() - start_time > deadline:
        return best_set
    
    # Strategy 2: Upper 2/3
    candidate2 = list(range(85, 257))
    repaired2 = _greedy_construct(candidate2, deadline)
    if len(repaired2) > len(best_set):
        best_set = repaired2
    
    if time.time() - start_time > deadline:
        return best_set
    
    # Strategy 3: Randomized subsets, biased towards high numbers
    for _ in range(20):
        if time.time() - start_time > deadline:
            break
        size = random.randint(120, 160)
        pool = list(range(64, 257))
        random.shuffle(pool)
        candidate3 = pool[:size]
        random.shuffle(candidate3)
        candidate3.sort()
        repaired3 = _greedy_construct(candidate3, deadline)
        if len(repaired3) > len(best_set):
            best_set = repaired3
    
    if time.time() - start_time > deadline:
        return best_set
    
    # Strategy 4: Local search on best found so far
    improved = _local_search_swap(best_set, max_iterations=50, deadline=deadline)
    if len(improved) > len(best_set):
        best_set = improved
    
    if time.time() - start_time > deadline:
        return best_set
    
    # Try removing small elements and seeing if we can add more large ones
    for min_val in [100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230]:
        if time.time() - start_time > deadline:
            break
        candidate4 = list(range(min_val, 257))
        repaired4 = _greedy_construct(candidate4, deadline)
        if len(repaired4) > len(best_set):
            best_set = repaired4
    
    return best_set
# EVOLVE-BLOCK-END
