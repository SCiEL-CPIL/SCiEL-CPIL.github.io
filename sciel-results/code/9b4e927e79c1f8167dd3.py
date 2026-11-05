import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # Leave 45s margin for safety
    
    # Target differences
    target_diffs = set(range(1, 31))
    
    # We'll use a sweep over diameter M, forcing M as the max element.
    # This directly minimizes the largest element.
    
    # For each M, we need to find a Sidon set S ⊆ {0, ..., M} with 0, M ∈ S,
    # |S| ≤ 32, all positive differences distinct, and 1..30 ⊆ differences.
    
    # We use backtracking. To make it efficient:
    # 1. Fix 0 and M.
    # 2. The differences from M to other elements are M - s for s in S \ {M}.
    # 3. We need to fill in the missing differences.
    
    # A key insight: if we have elements a < b, the difference b-a is used.
    # For a Sidon set, all differences are distinct.
    
    # Let's use a smarter backtracking:
    # - We build the set incrementally.
    # - We maintain used_diffs (set of differences already realized).
    # - We maintain found_diffs (subset of 1..30 already realized).
    # - We try to add elements in increasing order.
    
    # But the sweep over M is key. For each M, we check if a valid set exists.
    # The first M for which we find a valid set gives us the answer.
    
    # To speed up the backtracking for a given M:
    # - We can precompute which differences are "forced" by 0 and M.
    #   0 and M give difference M. So M must be in used_diffs.
    #   But M might be > 30, so it doesn't help with 1..30 directly.
    
    # - We can use constraint propagation:
    #   If a difference d in 1..30 is not yet realized, we need some pair (a,b) with b-a=d.
    #   This constrains future elements.
    
    # Let's implement a backtracking search for a given M.
    
    def solve_for_M(M):
        if time.time() > deadline:
            return None
        
        # Initial set: {0, M}
        # used_diffs: {M} (if M != 0)
        # found_diffs: {d for d in 1..30 if d == M} (usually empty since M >= 30)
        
        initial_set = [0, M]
        initial_used = set()
        if M > 0:
            initial_used.add(M)
        
        initial_found = set()
        for d in range(1, 31):
            if d in initial_used:
                initial_found.add(d)
        
        # If we already have all differences, return the set.
        if initial_found.issuperset(target_diffs):
            return initial_set
        
        # Backtracking
        # We'll try to add elements between 1 and M-1.
        # To be efficient, we can use the following strategy:
        # - At each step, determine which elements can still be added.
        # - Prioritize elements that realize more missing differences.
        
        def backtrack(current_set, used_diffs, found_diffs):
            if time.time() > deadline:
                return None
            
            # Check if we have a solution
            if found_diffs.issuperset(target_diffs):
                return list(current_set)
            
            if len(current_set) >= 32:
                return None
            
            # Determine candidates: elements x in {1, ..., M-1} not in current_set
            # such that adding x doesn't violate the Sidon property.
            
            # For efficiency, we can limit candidates to those that realize at least one new difference.
            # But sometimes we need "neutral" elements.
            
            # Let's try candidates in order of how many new target differences they realize.
            
            candidates = []
            for x in range(1, M):
                if x in current_set:
                    continue
                
                # Check if x can be added
                valid = True
                new_found = set(found_diffs)
                for y in current_set:
                    d = abs(x - y)
                    if d in used_diffs:
                        valid = False
                        break
                    if d in target_diffs:
                        new_found.add(d)
                
                if not valid:
                    continue
                
                # Count how many new target differences are realized
                new_count = len(new_found - found_diffs)
                candidates.append((new_count, x, new_found))
            
            # Sort by new_count descending, then by x ascending
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            # If no candidates, return None
            if not candidates:
                return None
            
            # Try each candidate
            for new_count, x, new_found in candidates:
                # If this candidate doesn't realize any new differences and we have many missing,
                # we might skip it? No, let's try all.
                
                # Add x
                new_set = list(current_set)
                new_set.append(x)
                new_set.sort()
                
                new_used = set(used_diffs)
                for y in current_set:
                    new_used.add(abs(x - y))
                
                result = backtrack(new_set, new_used, new_found)
                if result is not None:
                    return result
            
            return None
        
        return backtrack(initial_set, initial_used, initial_found)
    
    # Sweep M from 30 upwards
    # A lower bound: we need 30 distinct differences, so we need at least ~sqrt(30) elements? 
    # Actually, a Sidon set of size n has ~n^2/2 differences. So for 30 differences, n ~ 8.
    # But the diameter must be at least 30 (since difference 30 must exist).
    
    for M in range(30, 10000):
        if time.time() > deadline:
            break
        
        result = solve_for_M(M)
        if result is not None:
            return result
    
    # Fallback: greedy
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
