"""Sidon set construction: cover differences 1..30 with minimal max element."""
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # We need a Sidon set (all positive differences distinct) containing 0,
    # with every d in 1..30 appearing as a difference.
    # Minimize the largest element.
    
    # Strategy: backtracking search over sorted sets containing 0.
    # Prune aggressively using:
    # 1. Difference count bound: with k elements, at most k(k-1)/2 differences.
    # 2. If remaining elements can't cover remaining differences, prune.
    # 3. All differences must be distinct (Sidon property).
    
    TARGET_DIFFS = set(range(1, 31))
    
    best_solution = None
    best_max = float('inf')
    
    def check_and_update(s):
        nonlocal best_solution, best_max
        if not s:
            return
        mx = max(s)
        if mx >= best_max:
            return
        # Check all 30 differences present
        diffs = set()
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                diffs.add(s[j] - s[i])
        if TARGET_DIFFS.issubset(diffs):
            best_solution = list(s)
            best_max = mx
    
    def backtrack(current, next_min, used_diffs, remaining_needed):
        nonlocal best_solution, best_max, deadline
        
        if time.time() > deadline:
            return
        
        # If we've covered all required differences, check solution
        if not remaining_needed:
            check_and_update(current)
            return
        
        # Bound: with current length k, we can add at most (32-k) more elements.
        # Total possible differences with k+m elements: (k+m)(k+m-1)/2
        # We need at least len(remaining_needed) more distinct differences.
        k = len(current)
        max_additional = 32 - k
        total_possible = (k + max_additional) * (k + max_additional - 1) // 2
        # We already have k(k-1)/2 differences, so new ones available:
        new_diffs_available = total_possible - k * (k - 1) // 2
        if new_diffs_available < len(remaining_needed):
            return
        
        # Also: each new element x added to current set creates differences
        # x - c for each c in current. These must be distinct from used_diffs
        # and from each other (but they're automatically distinct since c are distinct).
        # The key constraint: x - c must not be in used_diffs for any c,
        # AND the new differences must not collide with each other (they can't
        # since c are distinct), AND must not collide with existing diffs.
        
        # Try adding next element
        # The next element must be >= next_min
        # Upper bound: we want to keep max small
        # If current max is m, and we need to cover diffs up to 30,
        # the next element x must satisfy: for some c in current, x - c is in remaining_needed
        # OR x - c creates a "waste" difference (not in 1..30) but that's OK as long
        # as we can still cover all required ones.
        
        # Candidate next elements: for each required diff d and each c in current,
        # x = c + d is a candidate that would cover d.
        # Also x could be any value >= next_min, but we should focus on useful ones.
        
        candidates = set()
        for c in current:
            for d in remaining_needed:
                x = c + d
                if x >= next_min and x < 10**6:
                    candidates.add(x)
        
        # Also consider x = next_min itself (might create useful diffs)
        candidates.add(next_min)
        
        # Sort candidates for deterministic search
        for x in sorted(candidates):
            if time.time() > deadline:
                return
            
            # Check Sidon property: x - c must not be in used_diffs for any c in current
            valid = True
            new_diffs = set()
            for c in current:
                d = x - c
                if d <= 0:
                    continue
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Check that new diffs don't collide with each other (they can't since c distinct)
            # But we need to check they don't collide with used_diffs (done above)
            
            # Update remaining_needed
            new_remaining = remaining_needed - new_diffs
            
            # Pruning: check if we can still cover new_remaining
            new_k = k + 1
            new_max_additional = 32 - new_k
            new_total_possible = (new_k + new_max_additional) * (new_k + new_max_additional - 1) // 2
            new_new_diffs_available = new_total_possible - new_k * (new_k - 1) // 2
            if new_new_diffs_available < len(new_remaining):
                continue
            
            # Additional pruning: the maximum difference we can still create
            # with remaining elements. If current max is x, and we can add up to 
            # new_max_additional more elements, the max possible difference is bounded.
            # Actually this is tricky. Let's use a simpler bound:
            # The largest required diff in new_remaining must be achievable.
            # With new elements all >= x+1 (since sorted), the max difference from 
            # element x to a future element y is y - x. But y can be large.
            # However, we're minimizing max, so let's track that.
            
            # Recurse
            backtrack(
                current + [x],
                x + 1,
                used_diffs | new_diffs,
                new_remaining
            )
            
            if best_max <= x:
                # No point trying larger x if we already found solution with max <= x
                # Actually we should continue to find smaller max solutions
                pass
    
    # Start with just [0]
    backtrack([0], 1, set(), TARGET_DIFFS.copy())
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: use the greedy approach from the seed
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
