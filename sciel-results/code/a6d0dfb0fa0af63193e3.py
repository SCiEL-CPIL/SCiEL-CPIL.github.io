import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Use bitset-based backtracking for speed
    # We search for smallest D such that a Sidon set in [0,D] covers all diffs 1..30
    
    for D in range(70, 200):
        if time.time() > deadline:
            break
        
        # Bitset approach:
        # - used_diffs: bitmask where bit d is set if difference d is already used
        # - For candidate x, conflict if any (x-y) or (y-x) for y in current set has bit set in used_diffs
        # - We track current set as a sorted list
        
        target_mask = (1 << 31) - 2  # bits 1..30 set
        used_diffs = 0  # bitmask of used differences
        current_set = [0]
        
        # Precompute: for each position, which differences it would create with existing elements
        # This is dynamic, so we compute on the fly
        
        def dfs(current_set, used_diffs, missing_mask, last_elem):
            if missing_mask == 0:
                return current_set[:]
            
            if time.time() > deadline:
                return None
            
            k = len(current_set)
            # We need to cover missing_mask differences
            # Each new element x can cover at most k differences (with existing elements)
            # Plus it creates differences with future elements
            
            # Try candidates from last_elem+1 to D
            # Prioritize candidates that cover more missing differences
            
            # Generate candidates and score them
            candidates = []
            for x in range(last_elem + 1, D + 1):
                # Check if x conflicts with current set
                new_diffs_mask = 0
                valid = True
                for y in current_set:
                    d = x - y  # x > y since x > last_elem >= all in current_set
                    if d > 30:
                        # Difference > 30, just check it's not already used
                        # But we only track diffs 1..30 in used_diffs
                        # Diffs > 30 don't matter for our constraint
                        pass
                    else:
                        bit = 1 << d
                        if used_diffs & bit:
                            valid = False
                            break
                        new_diffs_mask |= bit
                
                if not valid:
                    continue
                
                # Also need to check: does x-x' for any pair create a diff already used?
                # Wait, we only check x against existing elements. The Sidon condition requires
                # ALL pairwise differences to be distinct. So we need to check x-y for all y in current_set
                # AND ensure those differences haven't been used before.
                # That's what we did above.
                
                # Score: how many missing diffs does this cover?
                covered = new_diffs_mask & missing_mask
                score = bin(covered).count('1')
                
                candidates.append((score, x, new_diffs_mask))
            
            # Sort by score descending (greedy: cover most missing first)
            candidates.sort(key=lambda t: -t[0])
            
            for score, x, new_diffs_mask in candidates:
                if score == 0 and len(candidates) > 1:
                    # If this covers nothing and there are better options, skip?
                    # No, we still need to try it, but deprioritize
                    pass
                
                new_missing = missing_mask & ~new_diffs_mask
                new_used = used_diffs | new_diffs_mask
                
                res = dfs(current_set + [x], new_used, new_missing, x)
                if res is not None:
                    return res
            
            return None
        
        result = dfs([0], 0, target_mask, 0)
        if result is not None:
            return result
    
    # Fallback: greedy construction
    a = [0]
    def get_diffs(s):
        s = sorted(s)
        return set(s[i] - s[j] for i in range(len(s)) for j in range(i))
    
    for d in range(1, 31):
        if d in get_diffs(a):
            continue
        found = False
        for x in range(1, 2000):
            if x in a:
                continue
            trial = sorted(a + [x])
            diffs = get_diffs(trial)
            if len(diffs) == len(trial) * (len(trial) - 1) // 2:
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 100
            while True:
                trial = sorted(a + [y, y + d])
                diffs = get_diffs(trial)
                if len(diffs) == len(trial) * (len(trial) - 1) // 2:
                    a = trial
                    break
                y += 1
    return a
