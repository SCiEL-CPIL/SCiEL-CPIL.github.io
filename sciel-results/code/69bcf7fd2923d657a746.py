import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
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
    
    # DFS State: S (list), used_diffs (set), covered_mask (int)
    # We want to minimize max(S).
    
    # Heuristic: Order candidates to find a valid solution quickly.
    # We prioritize adding elements that cover many missing diffs.
    
    def dfs(S, used_diffs, covered_mask, depth):
        nonlocal best_max, best_set
        
        if time.time() - start_time > deadline:
            return
            
        if covered_mask == full_mask:
            current_max = max(S)
            if current_max < best_max:
                best_max = current_max
                best_set = S[:]
            return
            
        if depth >= 32:
            return
            
        current_max = max(S) if S else 0
        
        # Pruning 1: If current max is already >= best_max, we can't improve diameter
        # unless we haven't covered all diffs yet? 
        # Actually, if we are building the set, adding more elements increases max.
        # If current_max >= best_max, any extension will have max >= current_max >= best_max.
        # So we can prune if we are not already better? 
        # Wait, if we found a solution with max 100, and we are at depth 5 with max 100, 
        # we might still find a solution with max 100? No, we want strictly smaller.
        # But if we are at max 100 and need to add more, the final max will be > 100.
        # So if current_max >= best_max, we can prune.
        if current_max >= best_max:
            return
            
        # Pruning 2: Can we cover remaining diffs?
        missing_count = 30 - bin(covered_mask).count('1')
        steps_left = 32 - depth
        # Max new diffs per new element is 'depth' (distances to all existing elements).
        # Actually, max new diffs is steps_left * depth? No.
        # Each new element adds at most 'depth' new diffs.
        # So total potential new diffs = sum_{i=1}^{steps_left} (depth + i - 1)
        # This is an upper bound. If this < missing_count, prune.
        max_potential_new_diffs = 0
        for i in range(steps_left):
            max_potential_new_diffs += depth + i
        if max_potential_new_diffs < missing_count:
            return

        # Generate Candidates
        # We need to pick the next element x > current_max.
        # To minimize max, we should try small x first.
        # But we also need to cover missing diffs.
        
        missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        
        # Candidate set: 
        # 1. x = s + d for s in S, d in missing_diffs. These x's cover d.
        # 2. x = current_max + k for small k (to keep max low).
        
        candidates = set()
        
        # Add candidates derived from missing diffs
        for d in missing_diffs:
            for s in S:
                x = s + d
                if x > current_max:
                    candidates.add(x)
                    
        # Add small increments to escape local minima / find tight packing
        for k in range(1, 10):
            x = current_max + k
            if x > current_max:
                candidates.add(x)
                
        # Sort candidates by value (smaller is better for diameter)
        candidates = sorted(candidates)
        
        # Optional: Score candidates by how many new diffs they cover to find valid solutions faster?
        # For now, try smallest x first. If we find a valid solution, best_max drops, 
        # allowing us to prune larger branches.
        
        for x in candidates:
            if x >= best_max:
                break # Since sorted, no need to check larger
            
            # Check Sidon property
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
                
            new_S = S + [x]
            new_used_diffs = used_diffs.union(new_diffs)
            
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
                    
            dfs(new_S, new_used_diffs, new_mask, depth + 1)
            
            if time.time() - start_time > deadline:
                return

    # Start DFS
    # Initial state: S=[0], used_diffs=set(), mask=0
    dfs([0], set(), 0, 1)
    
    if best_set:
        # Local Search Polish
        # Try to reduce each element to the minimum possible value.
        S = sorted(best_set[:])
        
        def try_reduce(S, i):
            """Try to reduce S[i] to the smallest valid value > S[i-1]."""
            if i == 0:
                return 0 # 0 is fixed
            
            current_val = S[i]
            S_rest = S[:i] + S[i+1:]
            lower_bound = S[i-1] + 1
            
            # Try values from lower_bound to current_val - 1
            for new_val in range(lower_bound, current_val):
                S_new = S_rest + [new_val]
                S_new.sort()
                
                # Check Sidon
                diffs = set()
                is_sidon = True
                for a in range(len(S_new)):
                    for b in range(a+1, len(S_new)):
                        d = S_new[b] - S_new[a]
                        if d in diffs:
                            is_sidon = False
                            break
                        diffs.add(d)
                    if not is_sidon:
                        break
                
                if not is_sidon:
                    continue
                
                # Check Coverage
                if not target_diffs.issubset(diffs):
                    continue
                
                return new_val
            
            return None # No reduction possible

        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            S.sort()
            # Try to reduce from largest index down to 1 (index 0 is 0)
            for i in range(len(S) - 1, 0, -1):
                new_val = try_reduce(S, i)
                if new_val is not None:
                    S[i] = new_val
                    S.sort()
                    improved = True
                    break # Restart loop as S changed
        
        return S
    else:
        # Fallback if DFS fails (should not happen)
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
