import time
import random

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit

    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1

    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])

    def is_sidon(S):
        if not S:
            return True
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2

    # Known good solution: Sidon set with small diameter covering 1..30
    # We'll try to find something better, but have a fallback.
    
    best_max = float('inf')
    best_set = None

    # Strategy: Systematic backtracking with pruning
    # We build the set incrementally, always adding the smallest valid element
    # that covers the most missing differences.
    
    def backtrack(S, used_diffs, covered_mask, depth, max_depth):
        nonlocal best_max, best_set
        nonlocal start_time, deadline
        
        if time.time() - start_time > deadline:
            return False
            
        # Check if we've covered all differences
        if covered_mask == full_mask:
            current_max = max(S)
            if current_max < best_max:
                best_max = current_max
                best_set = list(S)
            return True
            
        # If we've reached max depth, check if this is a valid partial solution
        # (we might not have covered all diffs, but we can't add more)
        if depth >= max_depth:
            return False
            
        # Pruning: if current max is already >= best_max, no point continuing
        if max(S) >= best_max:
            return False
            
        # Generate candidates: for each missing difference d, and each s in S,
        # x = s + d is a candidate. Also try s - d if positive.
        missing = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        
        candidates = set()
        for d in missing:
            for s in S:
                x = s + d
                if x > 0:
                    candidates.add(x)
                x2 = s - d
                if x2 > 0:
                    candidates.add(x2)
        
        # Sort candidates by how many new differences they cover
        def candidate_score(x):
            if x in S:
                return -1
            # Check if x is valid (no duplicate differences)
            new_diffs = []
            for s in S:
                d = abs(x - s)
                if d in used_diffs:
                    return -1
                new_diffs.append(d)
            # Count how many of the missing differences are covered
            new_covered = 0
            for d in new_diffs:
                if 1 <= d <= 30 and not (covered_mask & (1 << (d - 1))):
                    new_covered += 1
            # Prefer candidates that cover more missing differences, then smaller x
            return (-new_covered, x)
        
        valid_candidates = []
        for x in candidates:
            if x <= 0:
                continue
            if x in S:
                continue
            # Check Sidon condition
            valid = True
            new_diffs = []
            for s in S:
                d = x - s
                if d < 0:
                    d = -d
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if not valid:
                continue
            valid_candidates.append(x)
        
        if not valid_candidates:
            return False
        
        # Sort by score
        valid_candidates.sort(key=candidate_score)
        
        # Limit candidates to avoid explosion
        max_candidates = 20
        for x in valid_candidates[:max_candidates]:
            # Compute new state
            new_S = S + [x]
            new_used_diffs = set(used_diffs)
            new_mask = covered_mask
            for s in S:
                d = x - s
                if d < 0:
                    d = -d
                new_used_diffs.add(d)
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
            
            # Recursive call
            if backtrack(new_S, new_used_diffs, new_mask, depth + 1, max_depth):
                if best_max < float('inf'):
                    # We found a solution, but keep searching for better
                    # unless we're running out of time
                    if time.time() - start_time > deadline - 5:
                        return True
            else:
                # No solution found in this branch
                pass
        
        return best_max < float('inf')

    # Start backtracking
    # Initial set: {0}
    S_init = [0]
    used_diffs_init = set()
    covered_mask_init = 0
    max_depth = 32  # At most 32 elements
    
    # Try different starting strategies
    # Strategy 1: Just start with {0}
    backtrack(S_init, used_diffs_init, covered_mask_init, 1, max_depth)
    
    # If we found a solution, return it
    if best_set is not None:
        return best_set
    
    # Fallback: Greedy approach
    S = [0]
    covered = set()
    used_diffs = set()
    
    for _ in range(31):  # Add up to 31 more elements
        if time.time() - start_time > deadline - 5:
            break
            
        missing = target_diffs - covered
        if not missing:
            break
            
        best_candidate = None
        best_score = -1
        
        for d in missing:
            for s in S:
                x = s + d
                if x <= 0 or x in S:
                    continue
                # Check Sidon
                valid = True
                new_diffs = []
                for s2 in S:
                    diff = x - s2
                    if diff < 0:
                        diff = -diff
                    if diff in used_diffs:
                        valid = False
                        break
                    new_diffs.append(diff)
                if not valid:
                    continue
                # Score: how many missing diffs covered
                new_covered = 0
                for diff in new_diffs:
                    if diff in missing:
                        new_covered += 1
                score = (new_covered, -x)
                if score > best_score:
                    best_score = score
                    best_candidate = x
        
        if best_candidate is not None:
            x = best_candidate
            S.append(x)
            S.sort()
            for s in S[:-1]:
                d = x - s
                used_diffs.add(d)
                if 1 <= d <= 30:
                    covered.add(d)
        else:
            # No good candidate, add a large number
            x = 2 * max(S) + 3
            while True:
                valid = True
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    break
                x += 1
            S.append(x)
            S.sort()
            for s in S[:-1]:
                d = x - s
                used_diffs.add(d)
                if 1 <= d <= 30:
                    covered.add(d)
    
    return S
