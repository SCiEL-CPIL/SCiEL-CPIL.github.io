import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # We know N=9 is the minimum size.
    # Lower bound for M:
    # Number of differences is N(N-1)/2.
    # For N=9, 36 differences. They must be distinct and <= M.
    # So M >= 36.
    # Also, we need to cover 1..30.
    
    target_diffs = set(range(1, 31))
    
    def is_valid_set(S):
        # S is a sorted list of ints
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
        
    def backtrack(S, M, used_diffs, missing_diffs):
        # S: sorted list of current elements
        # M: max allowed element
        # used_diffs: set of differences already present
        # missing_diffs: set of differences from 1..30 not yet covered
        
        if time.time() > deadline:
            return None
            
        if len(S) == 9:
            if not missing_diffs:
                return S
            return None
            
        # Heuristic: We need to cover missing_diffs.
        # The new element x will create |S| new differences.
        # If |missing_diffs| > |S| * (9 - len(S)) + (9 - len(S)) * (8 - len(S)) / 2 ... no, just simple bound.
        # Max new diffs from adding k elements is k*|S| + k(k-1)/2.
        # Here we add one at a time.
        
        # Try to find a candidate x
        # x must be > S[-1] and <= M
        # Also, if we want to cover a specific missing diff d, x could be S[i] + d.
        
        # Strategy: Generate candidate x's.
        # 1. x = S[-1] + 1 ... M
        # 2. For each missing diff d, and each s in S, x = s + d.
        
        candidates = set()
        # Add simple increments
        for x in range(S[-1] + 1, M + 1):
            candidates.add(x)
            
        # Add candidates derived from missing diffs
        for d in missing_diffs:
            for s in S:
                x = s + d
                if x > S[-1] and x <= M:
                    candidates.add(x)
                    
        # Also try x such that x - s = d for some s, d
        # This is covered above.
        
        # Sort candidates by how many missing diffs they cover (greedy heuristic)
        def score_candidate(x):
            new_diffs = set()
            valid = True
            for s in S:
                d = x - s
                if d > M: # Should not happen if x<=M and s>=0
                    valid = False
                    break
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                return -1
            # Count how many missing diffs are in new_diffs
            return len(new_diffs & missing_diffs)
            
        scored_candidates = []
        for x in candidates:
            sc = score_candidate(x)
            if sc >= 0:
                scored_candidates.append((sc, x))
                
        # Sort by score descending, then x ascending (smaller x usually better for minimizing M later, but here M is fixed)
        scored_candidates.sort(key=lambda t: (-t[0], t[1]))
        
        for sc, x in scored_candidates:
            # Check validity again (just in case)
            new_diffs = set()
            valid = True
            for s in S:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            new_used_diffs = used_diffs | new_diffs
            new_missing_diffs = missing_diffs - new_diffs
            
            res = backtrack(S + [x], M, new_used_diffs, new_missing_diffs)
            if res is not None:
                return res
                
        return None

    # Search for N=9
    # Lower bound M:
    # We need 36 distinct diffs. Max diff is M.
    # So M >= 36.
    # Also, we need to cover 1..30.
    
    for M in range(36, 100):
        if time.time() > deadline:
            break
            
        # Initial state: S=[0]
        # Used diffs: empty
        # Missing diffs: 1..30
        res = backtrack([0], M, set(), target_diffs)
        if res is not None:
            return res
            
    # If N=9 fails, try N=10
    # For N=10, min diffs = 45. M >= 45.
    for M in range(45, 150):
        if time.time() > deadline:
            break
            
        # Modify backtrack to target N=10
        def backtrack10(S, M, used_diffs, missing_diffs):
            if time.time() > deadline:
                return None
            if len(S) == 10:
                if not missing_diffs:
                    return S
                return None
                
            candidates = set()
            for x in range(S[-1] + 1, M + 1):
                candidates.add(x)
            for d in missing_diffs:
                for s in S:
                    x = s + d
                    if x > S[-1] and x <= M:
                        candidates.add(x)
                        
            def score_candidate(x):
                new_diffs = set()
                valid = True
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    return -1
                return len(new_diffs & missing_diffs)
                
            scored_candidates = []
            for x in candidates:
                sc = score_candidate(x)
                if sc >= 0:
                    scored_candidates.append((sc, x))
                    
            scored_candidates.sort(key=lambda t: (-t[0], t[1]))
            
            for sc, x in scored_candidates:
                new_diffs = set()
                valid = True
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue
                    
                new_used_diffs = used_diffs | new_diffs
                new_missing_diffs = missing_diffs - new_diffs
                
                res = backtrack10(S + [x], M, new_used_diffs, new_missing_diffs)
                if res is not None:
                    return res
            return None
            
        res = backtrack10([0], M, set(), target_diffs)
        if res is not None:
            return res
            
    # Fallback to a known valid set (greedy result) if search fails
    # Re-implementing a fast greedy just in case
    a = [0]
    def diffs(s):
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
