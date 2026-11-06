import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # wall-clock limit
    MARGIN = 5.0      # safety margin
    
    def time_left():
        return (start_time + DEADLINE - MARGIN) - time.time()
    
    best_result = None
    best_max = float('inf')
    
    # We need a Sidon set S containing 0, with all positive differences distinct,
    # and every d in 1..30 appearing as a difference. Minimize max(S).
    
    # Strategy: backtracking search over sorted sets containing 0.
    # Try sizes from 9 upward (9 elements -> 36 diffs, need 30 of them to be 1..30)
    
    def get_diffs(s):
        """Get all positive differences of sorted set s."""
        diffs = []
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                diffs.append(s[j] - s[i])
        return diffs
    
    def is_sidon(s):
        """Check if all positive differences are distinct."""
        diffs = get_diffs(s)
        return len(set(diffs)) == len(diffs)
    
    def covers_1_to_30(s):
        """Check if every d in 1..30 appears as a difference."""
        diffs = set(get_diffs(s))
        return all(d in diffs for d in range(1, 31))
    
    # Backtracking: build set element by element
    # State: current set (sorted), set of used differences, set of covered targets
    # We want to cover all of 1..30
    
    targets = set(range(1, 31))
    
    def backtrack(current_set, used_diffs, covered, max_allowed):
        nonlocal best_result, best_max
        
        if time_left() <= 0:
            return
        
        # Pruning: if current max already exceeds best, stop
        if current_set and max(current_set) >= best_max:
            return
        
        # Check if we've covered all targets
        if covered == targets:
            # We have a valid solution
            if len(current_set) <= 32:
                m = max(current_set)
                if m < best_max:
                    best_max = m
                    best_result = list(current_set)
            return
        
        # If we can't cover remaining targets with remaining capacity, prune
        # Remaining elements: at most 32 - len(current_set)
        remaining_slots = 32 - len(current_set)
        if remaining_slots <= 0:
            return
        
        # Upper bound on how many more differences we can add
        # Each new element x added to a set of size k adds k new differences
        # So max additional diffs = sum of (len(current_set) + i) for i in range(remaining_slots)
        # Actually, if we add m more elements, total new diffs = m * len(current_set) + m*(m-1)//2
        # But we also need the Sidon property, so this is an upper bound
        
        # Simple pruning: if remaining targets can't be covered, stop
        # Each new element can cover at most (len(current_set) + position) new targets
        # But this is complex; let's use a simpler bound
        
        # Try adding next element
        # The next element must be > current_set[-1] (if current_set is non-empty)
        # Start from current_set[-1] + 1 if current_set else 1
        # But we can also try smaller values if we haven't fixed the order
        
        # Actually, let's build the set in increasing order, always adding the next largest
        # Start with 0, then add elements in increasing order
        
        if not current_set:
            # Must include 0
            new_set = (0,)
            new_used = set()
            new_covered = set()
            backtrack(new_set, new_used, new_covered, max_allowed)
            return
        
        # Current set is sorted, last element is current_set[-1]
        # Try next element from current_set[-1]+1 up to max_allowed
        # But max_allowed is our best_max - 1 (we want to improve)
        
        # Bound the search: next element can't be too large
        # If we add element x, the differences it creates are x - s for s in current_set
        # These must all be in used_diffs (Sidon) and we want to cover targets
        
        # Heuristic: try values that cover uncovered targets
        # For each uncovered target d, if d - s is positive for some s in current_set,
        # then adding d - s would cover d (if not already used as a diff)
        
        # Collect candidate next elements
        candidates = set()
        
        # For each uncovered target d, for each s in current_set:
        # candidate = d + s (this would make d = candidate - s)
        # Also candidate = d - s (this would make d = s - candidate, but s < candidate so no)
        # So candidate = d + s for d in uncovered, s in current_set
        # But also we might need elements that don't directly cover a target but enable future coverage
        
        # Let's be more systematic: try a range of candidate values
        # Start from current_set[-1] + 1
        # Upper bound: we need to be able to cover remaining targets
        # The largest remaining target is max(targets - covered) if any
        # To cover target d, we need some pair (a,b) in the final set with b-a=d
        # If we add element x, it can create differences x-s for s in current_set
        # So x can be at most max(remaining_targets) + max(current_set)
        
        remaining_targets = targets - covered
        if not remaining_targets:
            return
        
        max_remaining_target = max(remaining_targets)
        # Upper bound for next element
        upper_bound = min(best_max - 1, max_remaining_target + current_set[-1])
        
        # But also, we need enough room to cover all remaining targets
        # This is complex; let's just try a reasonable range
        
        # Try candidates in order of how many new targets they cover
        # For efficiency, generate candidates and sort by coverage
        
        # Generate candidate values
        cand_set = set()
        for d in remaining_targets:
            for s in current_set:
                c = d + s
                if c > current_set[-1] and c <= upper_bound:
                    cand_set.add(c)
        
        # Also try sequential values for completeness (but limit the range)
        # If no candidates from targets, try sequential
        if not cand_set:
            for v in range(current_set[-1] + 1, min(upper_bound, current_set[-1] + 50) + 1):
                cand_set.add(v)
        
        # Sort candidates by how many new targets they cover (greedy heuristic)
        def coverage_score(c):
            score = 0
            for s in current_set:
                d = c - s
                if d in remaining_targets and d not in used_diffs:
                    score += 1
            return score
        
        sorted_candidates = sorted(cand_set, key=lambda c: (-coverage_score(c), c))
        
        # Limit the number of candidates to try per node for efficiency
        max_candidates = 20
        for c in sorted_candidates[:max_candidates]:
            if time_left() <= 0:
                return
            
            # Check Sidon: all differences c - s for s in current_set must not be in used_diffs
            new_diffs = []
            valid = True
            for s in current_set:
                d = c - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Check that new_diffs are all positive (they are, since c > s)
            # Update state
            new_used = used_diffs | set(new_diffs)
            new_covered = covered | (set(new_diffs) & targets)
            
            # Prune: if we've used too many differences (can't happen with Sidon, but check)
            # The total number of differences in the final set will be len(final)*(len(final)-1)//2
            # We need this to be at least 30, and the Sidon property means all are distinct
            
            # Recurse
            new_set = current_set + (c,)
            backtrack(new_set, new_used, new_covered, max_allowed)
    
    # Start the search
    # Try different max_allowed values, starting from a reasonable lower bound
    # Lower bound: we need 30 differences, with k elements we have k(k-1)/2 differences
    # k=9: 36 diffs, k=10: 45 diffs
    # The minimum possible max element: for a perfect difference set of order n, 
    # the diameter is n(n-1)-1. For n=6, that's 30. So a (6,30,1) difference set has diameter 30.
    # But we need a Sidon set, not a perfect difference set.
    
    # Let's try max_allowed from a reasonable lower bound
    # Lower bound: at least 30 (since we need difference 30, and 0 is in the set, 
    # so some element must be at least 30)
    
    # Actually, if 0 is in the set and 30 is a difference, then either there's an element 30,
    # or there are elements a,b with b-a=30 and a>0. In the latter case, b >= 30.
    # So max element >= 30.
    
    # Let's try to find a solution with max element close to 30
    # A (6,30,1) perfect difference set would have 6 elements and all differences 1..30.
    # Known: {0,1,4,10,12,28} is a (6,30,1) difference set? Let me check.
    # Differences: 1,4,10,12,28, 3,9,11,27, 6,8,24, 2,26, 18, 20, 16
    # That's only 15 differences for 6 elements (6*5/2=15). We need 30 differences.
    # So we need more elements.
    
    # With 9 elements: 36 differences. We need 30 of them to be 1..30.
    # So 6 of the 36 differences are "wasted" (not in 1..30).
    
    # Let's just run the backtracking with a reasonable max_allowed
    # Start with max_allowed = 60 and try to find a solution
    
    # Actually, let's be smarter. Let's try specific constructions first.
    
    # Known small Sidon sets that are difference bases:
    # Let's try to use a (k, v, 1) difference set if one exists for v=30
    # A (6,30,1) difference set exists: {0,1,4,10,12,28} but that only has 15 differences.
    # We need a set where ALL differences are distinct AND cover 1..30.
    
    # For a Sidon set of size k, we have k(k-1)/2 distinct differences.
    # We need these to include 1..30.
    # So k(k-1)/2 >= 30, k >= 9.
    
    # Let's try k=9: 36 differences, need 30 to be 1..30, 6 are >30 or duplicates (but Sidon means no duplicates, so 6 are >30).
    
    # Run backtracking
    backtrack((), set(), set(), 10**6)
    
    # If we found a solution, return it
    if best_result is not None:
        return best_result
    
    # Fallback: use the seed's greedy approach
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
