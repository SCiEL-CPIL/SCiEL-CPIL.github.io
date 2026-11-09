import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    target_diffs = set(range(1, 31))
    
    def solve(M, min_n, max_n):
        """
        Find a Sidon set in [0, M] of size between min_n and max_n
        that covers all target_diffs.
        """
        # We fix 0
        current_A = [0]
        current_diffs = set()
        
        # Pre-compute candidates for each step? No, dynamic.
        
        def backtrack(last_elem, curr_A, curr_diffs):
            if time.time() > deadline:
                return None
            
            # Check if we have covered all differences
            if target_diffs.issubset(curr_diffs):
                return curr_A
            
            # If we have reached max size, fail
            if len(curr_A) >= max_n:
                return None
            
            # Pruning: Check if it's possible to cover remaining diffs
            missing = target_diffs - curr_diffs
            remaining_slots = max_n - len(curr_A)
            # Max new diffs from remaining_slots elements added to set of size s
            # is remaining_slots * len(curr_A) + (remaining_slots * (remaining_slots - 1)) // 2
            # This is an upper bound on new distinct diffs.
            # But we only care about covering 'missing'.
            # If len(missing) is large, we need to check if capacity is enough.
            # This bound is loose, but helps.
            
            # Generate candidates
            candidates = []
            for x in range(last_elem + 1, M + 1):
                # Check Sidon property
                valid = True
                new_diffs_set = set(curr_diffs)
                for a in curr_A:
                    d = x - a
                    if d in new_diffs_set:
                        valid = False
                        break
                    new_diffs_set.add(d)
                
                if not valid:
                    continue
                
                # Score: how many missing diffs does this x cover?
                covered = 0
                for d in new_diffs_set - curr_diffs:
                    if d in missing:
                        covered += 1
                
                # Only consider if it covers at least 1 missing diff?
                # Or just keep all and sort.
                # To reduce branching, maybe only keep top K?
                # But we need to be careful not to miss a solution.
                # Let's keep all valid candidates and sort by coverage.
                candidates.append((covered, x, new_diffs_set))
            
            # Sort by coverage descending, then by x ascending (prefer smaller x to keep diameter small? 
            # Actually we are fixing M, so x doesn't affect M directly, but smaller x leaves more room for larger x later?
            # No, we are adding in increasing order. Smaller x is generally "safer" for future additions?
            # Let's just sort by coverage.
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            # Heuristic: If no candidates cover any missing diff, and we still have missing diffs, 
            # we might be stuck unless we add "filler" elements that don't cover missing diffs 
            # but allow future elements to. 
            # However, if an element doesn't cover any missing diff, it's "wasted" for the goal.
            # But it might be necessary for structure.
            # Let's iterate through all candidates.
            
            for covered, x, new_diffs_set in candidates:
                # Optional pruning: if covered == 0 and len(missing) > 0, 
                # maybe skip? No, it might be needed.
                
                result = backtrack(x, curr_A + [x], new_diffs_set)
                if result:
                    return result
            
            return None
        
        return backtrack(0, [0], set())

    # Lower bound for M is 44 (based on optimal Golomb ruler for n=9)
    # We try M from 44 upwards.
    # We try n from 9 upwards.
    
    found_solution = None
    
    # It is highly likely that n=9 is sufficient.
    # If n=9 fails for M=44, try M=45, etc.
    # If n=9 fails for all M, try n=10.
    
    # To be efficient, we can interleave M and n.
    # But M is the primary objective.
    # So for M in 44, 45, ...:
    #    for n in 9, 10, ...:
    #       if solve(M, n, n): return
    
    # However, solving for n=10 is harder than n=9.
    # So for a fixed M, try n=9 first.
    
    for M in range(44, 100):
        if time.time() > deadline:
            break
        
        # Try n=9 first
        res = solve(M, 9, 9)
        if res:
            found_solution = res
            break
        
        # If n=9 fails, try n=10
        res = solve(M, 10, 10)
        if res:
            found_solution = res
            break
            
        # Try n=11
        res = solve(M, 11, 11)
        if res:
            found_solution = res
            break

    if found_solution:
        return found_solution

    # Fallback: Greedy construction (less optimal but always valid if it works)
    a = [0]
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for d in range(1, 31):
        if d in get_diffs(sorted(a)):
            continue
        found = False
        candidates = set()
        for v in a:
            if v + d not in a:
                candidates.add(v + d)
            if v - d >= 0 and v - d not in a:
                candidates.add(v - d)
        
        for x in sorted(candidates):
            trial = sorted(a + [x])
            d_list = get_diffs(trial)
            if len(set(d_list)) == len(d_list):
                a = trial
                found = True
                break
        if not found:
            # Add two elements
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                d_list = get_diffs(trial)
                if len(set(d_list)) == len(d_list):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END
