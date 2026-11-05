import time
import sys

_START = None

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Target differences
    target_diffs = list(range(1, 31))
    target_set = set(target_diffs)
    
    # We search for the minimum D
    # Lower bound for D:
    # A Sidon set of size k has at most k(k-1)/2 differences.
    # We need 30 differences. k(k-1)/2 >= 30 => k(k-1) >= 60 => k=9 gives 36, k=8 gives 28.
    # So we need at least 9 elements.
    # The smallest Sidon set of size 9 containing 0?
    # Also, the largest difference must be at least 30.
    # Let's start D from 30.
    
    for D in range(30, 200):
        if time.time() - _START > 110:
            break
        
        # used_diffs[d] is 1 if difference d is already present
        # We use a list of size D+1 for speed
        used_diffs = bytearray(D + 1)
        
        # marks are the current elements in the set, sorted
        # We start with 0
        marks = [0]
        
        # Pre-compute target coverage status
        # We need to check if all 1..30 are covered.
        
        def is_complete(used):
            for d in target_diffs:
                if not used[d]:
                    return False
            return True

        def dfs(marks, used):
            if is_complete(used):
                return True
            
            # Pruning: Can we possibly cover all missing diffs?
            # Count missing
            missing = 0
            for d in target_diffs:
                if not used[d]:
                    missing += 1
            
            # Current size k
            k = len(marks)
            # Max new differences we can add with one element is k
            # If we have m elements left to add (we don't know how many, but we are limited by D)
            # Actually, we just try to add elements.
            
            # Candidate generation
            # We can add any x in [0, D] not in marks.
            # To reduce symmetry and search space, we can enforce an order?
            # No, the set is not ordered in a way that fixes positions easily, 
            # but we can assume we add elements in increasing order of value?
            # If we enforce that the next element added is larger than the last one added,
            # we might miss sets where we add a small number later?
            # No, a set is just a set. We can build it in any order.
            # However, to avoid permutations, we can enforce that we only add elements
            # larger than the current maximum? 
            # If we do that, we might miss a solution where a small element is added late?
            # No, if we start with {0}, and only add larger elements, we can form any set containing 0.
            # Because any set {0, a1, a2, ..., ak} with 0 < a1 < ... < ak can be built
            # by adding a1, then a2, etc.
            # So we can restrict candidates to x > max(marks).
            
            current_max = marks[-1] if marks else -1
            
            # Heuristic: Try candidates that cover the most missing differences first.
            # But calculating this for all candidates is expensive.
            # Simple heuristic: Try smaller x first? Or larger x first?
            # Usually, smaller x packs more numbers into the range, which is good for minimizing D.
            # But we are fixing D. We just want *a* solution.
            
            # Let's try candidates in increasing order.
            for x in range(current_max + 1, D + 1):
                if x in marks:
                    continue
                
                # Check conflicts
                conflict = False
                new_diffs = []
                for y in marks:
                    d = abs(x - y)
                    if d > D:
                        conflict = True
                        break
                    if used[d]:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if conflict:
                    continue
                
                # If no conflict, recurse
                # Mark new diffs
                for d in new_diffs:
                    used[d] = 1
                
                marks.append(x)
                
                if dfs(marks, used):
                    return True
                
                marks.pop()
                
                # Unmark new diffs
                for d in new_diffs:
                    used[d] = 0
            
            return False

        # Start DFS
        # To make it faster, we can add a time check inside
        start_time = time.time()
        if dfs(marks, used_diffs):
            return marks
        
    # Fallback if no solution found in time (should not happen for D < 200)
    # Return a known valid solution or a greedy one
    # Greedy fallback
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        placed = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                placed = True
                break
        if not placed:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
