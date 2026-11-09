import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin

    target_diffs = set(range(1, 31))
    target_mask_full = 0
    for d in range(1, 31):
        target_mask_full |= (1 << d)

    def popcount(x):
        return bin(x).count('1')

    def solve_for_M(M):
        # State: 
        # curr_A: tuple of elements (sorted)
        # curr_mask: bitmask of covered diffs in 1..30
        # used_diffs: set of ALL positive differences generated so far (for Sidon check)
        # last_elem: last element added
        
        def backtrack(curr_A, curr_mask, used_diffs, last_elem):
            if time.time() > deadline:
                return None
            
            if curr_mask == target_mask_full:
                return list(curr_A)
            
            # Pruning:
            # 1. If no more elements can be added (last_elem == M), fail.
            if last_elem == M:
                return None
            
            # 2. Check if remaining range can provide enough new differences.
            missing_count = 30 - popcount(curr_mask)
            
            # Max new elements we can add: M - last_elem
            # Let k = M - last_elem
            # Max new diffs = k * len(curr_A) + k * (k - 1) // 2
            # If missing_count > max_new_diffs, prune
            k_max = M - last_elem
            max_new_diffs = k_max * len(curr_A) + k_max * (k_max - 1) // 2
            if missing_count > max_new_diffs:
                return None
            
            # Heuristic: Try candidates x in [last_elem+1, M]
            # Prioritize x that cover more missing diffs?
            # Generate candidates and sort by potential score
            
            candidates = []
            for x in range(last_elem + 1, M + 1):
                # Check Sidon property
                valid = True
                new_diffs = []
                for a in curr_A:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Calculate score: number of new target diffs covered
                new_mask = curr_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << d)
                
                # Score is the number of bits set in (new_mask & ~curr_mask)
                score = popcount(new_mask & ~curr_mask)
                
                # We also prefer smaller x to keep diameter small? 
                # Since we are iterating M, smaller x is generally good for future flexibility.
                # But covering more targets is more important.
                # Let's sort by (-score, x)
                candidates.append((x, score, new_mask, new_diffs))
            
            if not candidates:
                return None
            
            # Sort candidates: highest score first, then smallest x
            candidates.sort(key=lambda item: (-item[1], item[0]))
            
            for x, score, new_mask, new_diffs in candidates:
                if time.time() > deadline:
                    return None
                
                # Update used_diffs
                new_used = set(used_diffs)
                new_used.update(new_diffs)
                
                # Recurse
                result = backtrack(curr_A + (x,), new_mask, new_used, x)
                if result:
                    return result
            
            return None
        
        # Start with 0
        return backtrack((0,), 0, set(), 0)

    # Iterate M upwards
    # Lower bound: 30.
    # For n=9, M is likely around 30-50.
    
    found_solution = None
    # Start from a reasonable lower bound.
    # Min diameter for Sidon set of size 9 is approx 30-40.
    # Let's start from 35.
    
    for M in range(35, 100):
        if time.time() > deadline:
            break
        
        res = solve_for_M(M)
        if res:
            found_solution = res
            break
    
    if found_solution:
        return found_solution

    # Fallback: Greedy construction
    # This is slower but guaranteed to find *a* solution if one exists with larger M.
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    # Greedy: add elements to cover missing diffs
    # We iterate d from 1 to 30. If d is not covered, try to add an element x such that x-a=d or a-x=d for some a in A.
    # To minimize max(A), we should try small x first.
    
    a = [0]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
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
            d_list = diffs(trial)
            if len(set(d_list)) == len(d_list):
                a = trial
                found = True
                break
        if not found:
            # Add two elements y, y+d far away
            base = 2 * max(a) + 10
            while True:
                y = base
                x = y + d
                trial = sorted(a + [y, x])
                d_list = diffs(trial)
                if len(set(d_list)) == len(d_list):
                    a = trial
                    break
                base += 1
    return a

# EVOLVE-BLOCK-END
