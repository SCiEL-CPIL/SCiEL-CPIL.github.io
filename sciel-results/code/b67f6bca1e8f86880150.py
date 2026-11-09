"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We need a Sidon set containing 0 with all differences 1..30 present.
    # Minimize the largest element.
    # 
    # A Sidon set (B2 sequence) has all positive differences distinct.
    # We need at most 32 elements in 0..10^6.
    #
    # Strategy: Use a backtracking search to find a small Sidon set that
    # covers all differences 1..30.
    #
    # Key insight: For a Sidon set of size n, the maximum element must be
    # at least ~n^2/4. For n=32, that's ~256. But we need to cover 30
    # specific differences.
    #
    # Let's use a smarter approach: 
    # - Start with {0}
    # - At each step, try to add elements that help cover missing differences
    # - Use forward checking: maintain a set of used differences
    # - When adding a new element x, it creates differences |x - y| for all y in current set
    # - All these differences must be new (not already used)
    #
    # To minimize the max element, we should prefer smaller candidates.
    #
    # Better approach: Since we know the structure of small Sidon sets,
    # let's try a systematic search.
    #
    # Actually, let's think about what minimum diameter is possible.
    # We need 30 distinct positive differences from 1..30 all present.
    # A Sidon set of size k has at most k*(k-1)/2 differences.
    # For k=9: 36 differences, enough for 30.
    # For k=8: 28 differences, not enough.
    # So we need at least 9 elements.
    #
    # Let's try to find a Sidon set of size 9-12 with small diameter.
    #
    # Known small Sidon sets:
    # {0,1,4,10,12,17,21,23,24,25,29,30} - let me check
    #
    # Let me use a backtracking approach with careful pruning.
    
    import sys
    sys.setrecursionlimit(10000)
    
    target_diffs = set(range(1, 31))
    
    def solve():
        # State: current set of elements (sorted), set of used differences
        # We'll use iterative deepening on the maximum element
        
        # Try increasing max_val from small to large
        # For each max_val, try to find a Sidon subset of [0, max_val] 
        # containing 0 with all diffs 1..30
        
        # Better: backtracking with element selection
        # Start with 0, then pick next elements
        
        used_diffs = set()
        elements = [0]
        
        # Precompute: for a given set of elements, what diffs are used
        # When adding x to elements, new diffs are |x - e| for e in elements
        
        def backtrack(elements, used_diffs, max_allowed):
            if len(elements) > 32:
                return None
            
            # Check if all target diffs are covered
            if target_diffs.issubset(used_diffs):
                return elements
            
            # If we've used too many elements, stop
            if len(elements) >= 32:
                return None
            
            # Determine which diffs are still missing
            missing = target_diffs - used_diffs
            if not missing:
                return elements
            
            # Generate candidate next elements
            # A new element x must create at least one missing diff
            # x - e = d or e - x = d for some e in elements, d in missing
            # So x = e + d or x = e - d for e in elements, d in missing
            # Also x >= 0 and x <= max_allowed and x not in elements
            
            candidates = set()
            for e in elements:
                for d in missing:
                    x1 = e + d
                    x2 = e - d
                    if 0 <= x1 <= max_allowed and x1 not in elements:
                        candidates.add(x1)
                    if 0 <= x2 <= max_allowed and x2 not in elements:
                        candidates.add(x2)
            
            if not candidates:
                return None
            
            # Sort candidates by value (prefer smaller for minimization)
            for x in sorted(candidates):
                # Check if adding x maintains Sidon property
                new_diffs = set()
                valid = True
                for e in elements:
                    diff = abs(x - e)
                    if diff in used_diffs:
                        valid = False
                        break
                    new_diffs.add(diff)
                
                if not valid:
                    continue
                
                # Also check that new_diffs don't conflict with each other
                # (they can't since they're all |x-e| for different e, 
                #  but two different e1, e2 could give same |x-e1|=|x-e2|
                #  only if e1=e2, so no issue)
                
                new_used = used_diffs | new_diffs
                result = backtrack(elements + [x], new_used, max_allowed)
                if result is not None:
                    return result
            
            return None
        
        # Iterative deepening on max element
        # Lower bound: we need 30 diffs, with k elements we have k*(k-1)/2 diffs
        # Min k = 9. For k=9, min max element?
        # Let's try max_val from 30 upward
        
        for max_val in range(30, 2000):
            result = backtrack([0], set(), max_val)
            if result is not None:
                return sorted(result)
        
        # Fallback: use the greedy approach
        a = [0]
        def diffs(s):
            s = sorted(s)
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
    
    return solve()
# EVOLVE-BLOCK-END
