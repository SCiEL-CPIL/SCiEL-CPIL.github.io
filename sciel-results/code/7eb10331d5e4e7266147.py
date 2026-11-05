"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    target_bits = 30
    full_mask = (1 << target_bits) - 1
    solution = None
    
    def dfs_find(D, current_set, used_diffs, covered_mask, next_start):
        nonlocal solution
        
        if covered_mask == full_mask:
            solution = current_set
            return True
        
        if time.time() > deadline:
            return False
            
        if len(current_set) == 32:
            return False
            
        # Candidates
        candidates = []
        # We can also add a pruning: 
        # Max elements left = 32 - len(current_set)
        # Max new diffs = sum_{i=1}^{max_left} (len(current_set) + i - 1)
        # This is not a tight bound on coverage.
        
        for x in range(next_start, D):
            new_diffs = []
            valid = True
            for a in current_set:
                d = x - a
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
                    
            # Prune if no new coverage
            if new_mask == covered_mask:
                continue
                
            new_bits = bin(new_mask ^ covered_mask).count('1')
            candidates.append((new_bits, x, new_diffs, new_mask))
            
        # Sort by coverage (greedy)
        candidates.sort(key=lambda t: -t[0])
        
        for _, x, new_diffs, new_mask in candidates:
            new_set = current_set + [x]
            new_used = used_diffs.union(new_diffs)
            
            if dfs_find(D, new_set, new_used, new_mask, x + 1):
                return True
                
        return False

    # Search for minimal D
    # Min size 9 => Min D 8
    for D in range(8, 1000): 
        if time.time() > deadline:
            break
            
        init_mask = 0
        if 1 <= D <= 30:
            init_mask |= (1 << (D - 1))
            
        init_set = [0, D]
        init_used = {D}
        
        # If D is very small, we might not be able to fit 9 elements
        # But dfs will handle it.
        
        if dfs_find(D, init_set, init_used, init_mask, 1):
            return solution
            
    # Fallback: Seed logic
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

# EVOLVE-BLOCK-END
