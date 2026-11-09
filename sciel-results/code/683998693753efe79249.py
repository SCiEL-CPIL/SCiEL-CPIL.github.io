import time
import sys

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    best_max = float('inf')
    best_set = None
    
    # We will try to find the best set for each size n from 9 to 32
    # Start with n=9, then 10, etc.
    # If we find a solution for n, we can use its max as the upper bound for subsequent searches.
    
    def dfs(S, used_diffs, covered_mask, n, current_max):
        nonlocal best_max, best_set
        
        if time.time() - start_time > deadline:
            return
        
        if len(S) == n:
            if covered_mask == full_mask:
                if current_max < best_max:
                    best_max = current_max
                    best_set = list(S)
            return
        
        if current_max >= best_max:
            return
            
        # Generate candidates
        # We want to add x > current_max
        # x - s must not be in used_diffs for any s in S
        # It's good if x - s covers a missing difference
        
        missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        
        candidates = set()
        # Candidates derived from covering missing differences
        for d in missing_diffs:
            for s in S:
                x = s + d
                if x > current_max:
                    candidates.add(x)
        
        # We might also need candidates that don't immediately cover a missing diff?
        # If we are stuck, we might need to "jump" ahead.
        # But for minimal max, we usually want to fill gaps.
        # Let's also add some "safe" candidates just in case?
        # For now, let's stick to candidates that cover missing diffs.
        # If this fails, we might need to expand.
        
        # Sort candidates by value
        sorted_candidates = sorted(candidates)
        
        for x in sorted_candidates:
            # Check if x is valid
            valid = True
            new_diffs = []
            for s in S:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            new_used_diffs = used_diffs.union(new_diffs)
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
            
            new_S = S + (x,)
            new_max = x
            
            dfs(new_S, new_used_diffs, new_mask, n, new_max)
            
            if time.time() - start_time > deadline:
                return

    # Try sizes from 9 to 32
    for n in range(9, 33):
        S_init = (0,)
        used_diffs_init = frozenset()
        covered_mask_init = 0
        dfs(S_init, used_diffs_init, covered_mask_init, n, 0)
        
        # If we found a solution, we can break if we are confident no smaller n works?
        # No, we want the global min max.
        # But usually n=9 is the smallest size that can work (28 diffs < 30).
        # So n=9 is the primary candidate.
        # If n=9 finds a solution with max M, can n=10 find a solution with max < M?
        # It's possible but unlikely.
        # We can continue checking, but we can prune if current_max >= best_max.
        
        if time.time() - start_time > deadline:
            break

    if best_set:
        return best_set
    else:
        # Fallback: simple greedy
        S = [0]
        covered = set()
        while not covered.issuperset(target_diffs) and len(S) <= 32:
            missing = target_diffs - covered
            found = False
            for d in sorted(missing):
                for s in S:
                    c = s + d
                    if c in S:
                        continue
                    new_S = S + [c]
                    new_S.sort()
                    # Check Sidon
                    diffs = set(y - x for i, x in enumerate(new_S) for y in new_S[i+1:])
                    if len(diffs) == len(new_S) * (len(new_S) - 1) // 2:
                        S = new_S
                        covered = diffs
                        found = True
                        break
                if found:
                    break
            if not found:
                # Add a large number to avoid collision
                c = 2 * max(S) + 3
                while True:
                    new_S = S + [c]
                    new_S.sort()
                    diffs = set(y - x for i, x in enumerate(new_S) for y in new_S[i+1:])
                    if len(diffs) == len(new_S) * (len(new_S) - 1) // 2:
                        S = new_S
                        covered = diffs
                        break
                    c += 1
        return S
