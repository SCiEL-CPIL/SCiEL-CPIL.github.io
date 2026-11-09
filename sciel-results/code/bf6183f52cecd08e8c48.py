import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    
    def get_diffs(S):
        """Get all positive differences from sorted list S"""
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
    
    def is_sidon(S):
        """Check if S is a Sidon set"""
        n = len(S)
        if n < 2:
            return True
        diffs = get_diffs(S)
        return len(diffs) == n * (n - 1) // 2
    
    def covers_all(S):
        """Check if all target differences 1..30 are present"""
        diffs = get_diffs(S)
        return target_diffs.issubset(diffs)
    
    def valid_solution(S):
        """Check if S is a valid solution"""
        if len(S) < 2 or len(S) > 32:
            return False
        if 0 not in S:
            return False
        if max(S) > 10**6:
            return False
        if not is_sidon(S):
            return False
        if not covers_all(S):
            return False
        return True
    
    def backtrack_search(D, min_size=9):
        """
        Backtracking search for a Sidon set in [0, D] containing 0,
        with all differences 1..30 covered.
        Uses strong pruning based on difference coverage.
        """
        target = set(range(1, 31))
        
        def dfs(current_set, used_diffs, covered, last_elem, depth):
            nonlocal deadline, start_time
            
            if time.time() > deadline:
                return None
            
            # Check if we've covered all target differences
            if covered == target:
                return current_set[:]
            
            # Pruning: can we still cover all missing differences?
            missing = target - covered
            if not missing:
                return current_set[:]
            
            # Current set size
            k = len(current_set)
            
            # Max elements we can add: at most 32 - k
            max_add = 32 - k
            
            # Each new element can cover at most k new differences (with existing elements)
            # plus differences with other new elements
            # Conservative bound: each new element covers at most k + (remaining new elements) differences
            # But we need to cover len(missing) differences
            # Simple bound: if even with max_add elements we can't cover enough, prune
            # Each element can contribute at most k differences with existing + some with new
            # Very conservative: each element contributes at most k new target diffs
            if len(missing) > max_add * k:
                return None
            
            # Try candidates: from last_elem + 1 to D
            # Prioritize candidates that cover many missing differences
            candidates = []
            for x in range(last_elem + 1, D + 1):
                new_diffs = set()
                valid = True
                for y in current_set:
                    d = x - y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Count how many missing diffs this covers
                new_covered_count = len(new_diffs & missing)
                
                if new_covered_count > 0:
                    candidates.append((new_covered_count, x, new_diffs))
            
            # Sort by coverage (descending), then by value (ascending)
            candidates.sort(key=lambda c: (-c[0], c[1]))
            
            # Limit candidates to keep search manageable
            max_candidates = 50
            if len(candidates) > max_candidates:
                candidates = candidates[:max_candidates]
            
            for _, x, new_diffs in candidates:
                new_covered = covered | (new_diffs & target)
                new_used = used_diffs | new_diffs
                new_set = current_set + [x]
                
                result = dfs(new_set, new_used, new_covered, x, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        return dfs([0], set(), set(), 0, 0)
    
    def local_search_optimize(S):
        """
        Local search to minimize max element while maintaining validity.
        Tries to replace each element with a smaller value.
        """
        if S is None:
            return None
        
        best_S = sorted(S)
        best_max = max(best_S)
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            improved = False
            
            # Try to reduce each element (except 0)
            S_sorted = sorted(best_S)
            n = len(S_sorted)
            
            for i in range(n - 1, 0, -1):  # Start from largest
                if time.time() > deadline - 5:
                    break
                
                s_i = S_sorted[i]
                S_rest = S_sorted[:i] + S_sorted[i+1:]
                
                # Try replacing s_i with smaller values
                # Focus on values that are close to current but smaller
                trial_range = min(s_i - 1, 30)
                if trial_range < 1:
                    continue
                
                trials = set()
                for offset in range(1, trial_range + 1):
                    trials.add(s_i - offset)
                
                # Also try some random smaller values
                lower = S_sorted[i-1] + 1 if i > 0 else 1
                if s_i > lower + 10:
                    for _ in range(20):
                        trials.add(lower + (s_i - 1 - lower) * (hash(str(s_i) + str(i) + str(no_improve_count)) % 1000) // 1000)
                
                for new_val in sorted(trials, reverse=True):
                    if new_val < lower or new_val >= s_i:
                        continue
                    
                    S_new = S_rest + [new_val]
                    S_new.sort()
                    
                    # Check if valid
                    if not is_sidon(S_new):
                        continue
                    if not covers_all(S_new):
                        continue
                    
                    new_max = max(S_new)
                    if new_max < best_max:
                        best_S = S_new
                        best_max = new_max
                        improved = True
                        break
                
                if improved:
                    break  # Restart with new best
            
            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
        
        return best_S
    
    # Main execution: try increasing D values with backtracking
    best_result = None
    best_score = float('inf')
    
    # Start from a reasonable lower bound
    # For a Sidon set of size k covering differences 1..30, 
    # the diameter must be at least ~sqrt(k*30) or so
    # Try D from 60 upwards
    for D in range(60, 200):
        if time.time() > deadline - 20:
            break
        
        result = backtrack_search(D)
        if result is not None:
            if valid_solution(result):
                score = max(result)
                if score < best_score:
                    best_score = score
                    best_result = result[:]
                    # Try to optimize further
                    optimized = local_search_optimize(best_result)
                    if optimized is not None and valid_solution(optimized):
                        opt_score = max(optimized)
                        if opt_score < best_score:
                            best_score = opt_score
                            best_result = optimized[:]
                    break  # Found a solution, stop searching larger D
    
    # If backtracking didn't find anything, use a greedy fallback
    if best_result is None:
        S = [0]
        covered = set()
        used_diffs = set()
        
        for step in range(32):
            if time.time() > deadline - 10:
                break
            
            missing = target_diffs - covered
            
            if not missing:
                break
            
            best_x = None
            best_coverage = -1
            
            # Try to find x that covers most missing differences
            for x in range(1, 500):
                new_diffs = set()
                valid = True
                for y in S:
                    d = abs(x - y)
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                coverage = len(new_diffs & missing)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_x = x
            
            if best_x is None:
                # Try larger values
                found = False
                base = max(S) + 1
                for offset in range(1, 100):
                    x = base + offset
                    new_diffs = set()
                    valid = True
                    for y in S:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        S.append(x)
                        S.sort()
                        covered = get_diffs(S)
                        used_diffs = get_diffs(S)
                        found = True
                        break
                
                if not found:
                    break
            else:
                S.append(best_x)
                S.sort()
                covered = get_diffs(S)
                used_diffs = get_diffs(S)
        
        if valid_solution(S):
            best_result = S
            best_score = max(S)
    
    # Final polish
    if best_result is not None:
        final = local_search_optimize(best_result)
        if final is not None and valid_solution(final):
            if max(final) < max(best_result):
                best_result = final
    
    # Ensure we return a valid set
    if best_result is None or not valid_solution(best_result):
        # Last resort: build a guaranteed valid set
        S = [0]
        covered = set()
        used_diffs = set()
        for d in range(1, 31):
            if d in covered:
                continue
            found = False
            for x in range(1, 2000):
                if x in S:
                    continue
                trial = sorted(S + [x])
                new_diffs = set()
                valid = True
                for i in range(len(trial)):
                    for j in range(i+1, len(trial)):
                        diff = trial[j] - trial[i]
                        if diff in new_diffs:
                            valid = False
                            break
                        new_diffs.add(diff)
                    if not valid:
                        break
                if valid:
                    S = trial
                    covered = new_diffs
                    used_diffs = new_diffs
                    found = True
                    break
            if not found:
                # Add a large element to force coverage
                y = 2 * max(S) + 3 * d + 10
                while True:
                    trial = sorted(S + [y])
                    new_diffs = set()
                    valid = True
                    for i in range(len(trial)):
                        for j in range(i+1, len(trial)):
                            diff = trial[j] - trial[i]
                            if diff in new_diffs:
                                valid = False
                                break
                            new_diffs.add(diff)
                        if not valid:
                            break
                    if valid:
                        S = trial
                        covered = new_diffs
                        used_diffs = new_diffs
                        break
                    y += 1
        
        best_result = S
    
    return best_result
