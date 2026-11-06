"""Combined approach: backtracking with constraint-guided candidate generation."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    target_mask = (1 << 30) - 1  # Bits 0-29 represent diffs 1-30
    
    best_solution = None
    best_val = float('inf')
    
    # Greedy baseline to ensure we have something valid
    def _greedy():
        a = [0]
        used = set()
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            # Try to add a single element that covers d
            found = False
            for base in a:
                for cand in (base + d, base - d):
                    if cand < 0 or cand in a:
                        continue
                    new_diffs = set()
                    ok = True
                    for y in a:
                        diff = abs(cand - y)
                        if diff in used or diff in new_diffs:
                            ok = False
                            break
                        new_diffs.add(diff)
                    if ok:
                        a.append(cand)
                        a.sort()
                        used |= new_diffs
                        covered |= (new_diffs & target_diffs)
                        found = True
                        break
                if found:
                    break
            if not found:
                # Add two elements to cover d
                y = max(a) + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    new_used = set()
                    ok = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            diff = trial[j] - trial[i]
                            if diff in new_used:
                                ok = False
                                break
                            new_used.add(diff)
                        if not ok:
                            break
                    if ok:
                        a = trial
                        used = new_used
                        covered |= (new_used & target_diffs)
                        break
                    y += 1
        return a
    
    baseline = _greedy()
    if baseline and 0 in baseline and len(baseline) <= 32:
        best_solution = baseline
        best_val = max(baseline)
    
    # Backtracking search with constraint-guided candidates
    def _backtrack(M, time_limit):
        """Search for Sidon set in [0, M] covering 1..30."""
        result = [None]
        
        def dfs(current_set, used_diffs, covered_mask, depth):
            if time.time() > time_limit:
                return False
            
            if covered_mask == target_mask:
                result[0] = current_set[:]
                return True
            
            if len(current_set) >= 32:
                return False
            
            last = current_set[-1]
            
            # Generate candidates: for each remaining diff d, try x = a + d
            # for each a in current_set where x > last and x <= M
            remaining = [d for d in range(1, 31) if not (covered_mask & (1 << (d-1)))]
            
            candidate_set = set()
            for d in remaining:
                for a in current_set:
                    x = a + d
                    if x > last and x <= M:
                        candidate_set.add(x)
            
            if not candidate_set:
                return False
            
            # Score candidates by how many remaining diffs they cover
            scored = []
            for x in candidate_set:
                new_diffs = []
                conflict = False
                for a in current_set:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.append(d)
                
                if conflict:
                    continue
                
                # Count new coverage
                new_mask = 0
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d-1))
                
                improvement = bin(new_mask & ~covered_mask).count('1')
                if improvement == 0:
                    continue
                
                scored.append((-improvement, x, new_diffs, new_mask))
            
            if not scored:
                return False
            
            # Sort by improvement (descending), then by value (ascending)
            scored.sort()
            
            for _, x, new_diffs, new_mask in scored:
                new_used = used_diffs | set(new_diffs)
                new_covered = covered_mask | new_mask
                new_set = current_set + [x]
                
                if dfs(new_set, new_used, new_covered, depth + 1):
                    return True
            
            return False
        
        if dfs([0], set(), 0, 0):
            return result[0]
        return None
    
    # Try increasing M values
    lower_bound = 35
    upper_bound = min(best_val, 200)
    
    for M in range(lower_bound, upper_bound + 1):
        if time.time() > deadline - 5.0:
            break
        
        # Give each M a time slice
        time_slice = 8.0 if M < 50 else 5.0
        t_start = time.time()
        limit = t_start + time_slice
        
        sol = _backtrack(M, limit)
        if sol is not None:
            # Verify
            diffs = set()
            valid = True
            for i in range(len(sol)):
                for j in range(i + 1, len(sol)):
                    d = sol[j] - sol[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            
            if valid and all(d in diffs for d in range(1, 31)) and len(sol) <= 32:
                best_solution = sol
                best_val = max(sol)
                break  # This is the minimum M, so we're done
    
    return best_solution
# EVOLVE-BLOCK-END
