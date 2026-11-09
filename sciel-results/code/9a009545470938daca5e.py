import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill
    
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

    def covers_all(S):
        diffs = get_diffs(S)
        return target_diffs.issubset(diffs)

    best_max = float('inf')
    best_set = None
    
    # Beam Search Parameters
    beam_width = 2000
    max_candidates_per_state = 10
    
    # State: (max_val, num_missing, S, used_diffs, covered_mask)
    S_init = (0,)
    used_diffs_init = set()
    covered_mask_init = 0
    max_val_init = 0
    
    beam = [(max_val_init, 30, S_init, used_diffs_init, covered_mask_init)]
    
    rng = random.Random(42)

    def get_candidates(S, used_diffs, covered_mask):
        candidates = set()
        # Generate candidates that cover missing differences
        # To reduce candidate count, we don't generate for ALL missing diffs if there are too many
        # But we want to cover them.
        missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        
        # Shuffle missing diffs to add randomness and avoid bias
        rng.shuffle(missing_diffs)
        
        # Limit the number of missing diffs we consider for candidate generation per state
        # to keep candidate set small
        # If we have 30 missing, and 10 elements in S, that's 300 candidates.
        # We will filter them.
        
        for d in missing_diffs:
            for s in S:
                x = s + d
                candidates.add(x)
        
        # Add some random candidates to escape local minima
        if S:
            last = max(S)
            for _ in range(5):
                rand_x = last + rng.randint(1, 20)
                if rand_x <= 10**6:
                    candidates.add(rand_x)
        
        valid_candidates = []
        S_list = list(S)
        last = S_list[-1] if S_list else -1
        
        for x in candidates:
            if x <= last:
                continue
            if x > 10**6:
                continue
            
            new_diffs = []
            valid = True
            for s in S_list:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
            
            new_missing = 30 - bin(new_mask).count('1')
            # Score: prefer smaller x (to keep max low), then more coverage
            valid_candidates.append((x, new_mask, new_diffs, new_missing))
            
        # Sort and limit
        # Primary key: x (smaller is better for minimizing max)
        # Secondary key: new_missing (fewer is better)
        valid_candidates.sort(key=lambda item: (item[0], item[3]))
        return valid_candidates[:max_candidates_per_state]

    for step in range(32):
        if time.time() - start_time > deadline:
            break
            
        new_beam = []
        for max_val, num_missing, S, used_diffs, covered_mask in beam:
            if time.time() - start_time > deadline:
                break
                
            if covered_mask == full_mask:
                if max_val < best_max:
                    best_max = max_val
                    best_set = list(S)
                continue
            
            candidates = get_candidates(S, used_diffs, covered_mask)
            for x, new_mask, new_diffs, new_missing in candidates:
                new_S = S + (x,)
                new_used_diffs = used_diffs.union(new_diffs)
                new_max = x
                
                new_beam.append((new_max, new_missing, new_S, new_used_diffs, new_mask))
                
                if time.time() - start_time > deadline:
                    break
                    
        if not new_beam:
            break
            
        # Sort by max_val primarily, then num_missing
        new_beam.sort(key=lambda item: (item[0], item[1]))
        
        seen = set()
        unique_beam = []
        for state in new_beam:
            s_tuple = state[2]
            if s_tuple not in seen:
                seen.add(s_tuple)
                unique_beam.append(state)
            if len(unique_beam) >= beam_width:
                break
                
        beam = unique_beam
        
        if best_max < float('inf'):
            beam = [s for s in beam if s[0] < best_max]
            if not beam:
                break

    if best_set:
        # Local Search Polish
        S = best_set
        # Try to reduce max element by swapping
        # We iterate from the largest element to the smallest
        # For each element s_i (i > 0), try to replace with s_j < s_i
        # such that the set remains Sidon and covers 1..30
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            # Sort S to ensure we process largest first? 
            # The set S is sorted in the beam search (since we add x > last).
            # So S[-1] is the max.
            
            # We try to improve each element, starting from the largest
            for i in range(len(S) - 1, 0, -1):
                if (time.time() - start_time) > deadline:
                    break
                    
                s_i = S[i]
                # Try to replace s_i with a smaller value
                # We don't need to try all values down to 1, just check if a smaller valid one exists
                # But finding it is hard. 
                # Heuristic: Try values in the gap between S[i-1] and S[i]
                # Or just try a few random smaller values?
                
                # Better: Check if we can reduce S[i] by 1?
                # If S[i] - 1 is valid, do it.
                # This is very slow if we have to check many times.
                
                # Let's try a limited local search:
                # For the current largest element S[-1], try to replace it with a smaller number.
                # We generate candidates for replacement:
                # Any number x < S[-1] such that S' = (S \ {S[-1]}) U {x} is valid.
                
                # This is expensive.
                # Given the time constraint, we might skip complex local search
                # and rely on the beam search finding a good solution.
                
                # However, a simple "greedy swap" might help.
                # Let's try to reduce the max element by checking if max-1 is valid.
                # If yes, replace and restart loop.
                if i == len(S) - 1:
                    new_val = s_i - 1
                    if new_val > S[i-1]: # Must be distinct and sorted
                        # Check if new_val creates a conflict
                        S_rest = S[:i] + S[i+1:]
                        S_new = S_rest + [new_val]
                        S_new.sort()
                        if is_sidon(S_new) and covers_all(S_new):
                            S = S_new
                            improved = True
                            break # Restart the outer while loop with new S

        return S
    else:
        # Fallback
        S = [0]
        covered = set()
        while not covered.issuperset(target_diffs) and len(S) <= 32:
            missing = target_diffs - covered
            found = False
            for d in missing:
                for s in S:
                    c = s + d
                    if c in S or c < 0:
                        continue
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        found = True
                        break
                if found:
                    break
            if not found:
                c = 2 * max(S) + 3
                while True:
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        break
                    c += 1
        return S
