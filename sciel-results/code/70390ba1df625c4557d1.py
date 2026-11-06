import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    target_diffs = set(range(1, 31))
    
    def get_diffs(s):
        return set(y - x for i, x in enumerate(s) for y in s[i+1:])
    
    def is_sidon(s):
        # s is sorted
        diffs = []
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                d = s[j] - s[i]
                if d in diffs:
                    return False
                diffs.append(d)
        return True
        
    def covers_target(s):
        diffs = get_diffs(s)
        return target_diffs.issubset(diffs)
        
    def backtrack(s, current_diffs, target_covered):
        """
        s: current sorted list
        current_diffs: set of differences generated so far
        target_covered: set of target diffs (1..30) covered so far
        """
        if time.time() > deadline:
            return None
            
        if covers_target(s) and is_sidon(s):
            return s
            
        if len(s) >= 12: # Max size constraint or just a reasonable upper bound for this search
            return None
            
        # Pruning: if we can't possibly cover all targets
        # Max new diffs from adding k more elements to current m elements is k*(m+k-1)/2? No.
        # Just check if target_covered can be completed.
        
        last = s[-1]
        # Try adding next element
        # We can bound the search space.
        # If we are at size n, min max element is roughly n^2.
        # We iterate N outside, so here we just pick the next element > last.
        
        # Heuristic: Try smaller elements first to keep diameter small?
        # Actually, we are searching for a specific N in the outer loop.
        # So we should only pick elements <= N.
        
        return None

    # We will search for minimal N
    # Lower bound for N:
    # For n=9, min N is 44.
    # For n=10, min N is 55.
    
    best_set = None
    
    # Try sizes 9, 10, 11
    for n in range(9, 12):
        # Estimate lower bound for N
        # A005282 values: 9->44, 10->55, 11->68
        lb = {9: 44, 10: 55, 11: 68}.get(n, n*n)
        
        for N in range(lb, 100):
            if time.time() > deadline:
                break
                
            # Backtracking search for size n in [0, N]
            # Start with [0]
            s = [0]
            diffs = set()
            covered = set()
            
            def search(idx, s, diffs, covered):
                if time.time() > deadline:
                    return None
                
                if len(s) == n:
                    if target_diffs.issubset(diffs):
                        return list(s)
                    return None
                
                # Pruning
                # Remaining elements to pick: rem = n - len(s)
                # Max new diffs: rem * (len(s) + rem - 1) / 2 + ... 
                # Actually, just check if we can cover missing targets.
                missing = target_diffs - covered
                if not missing:
                    # We covered all targets, just need to fill up to size n?
                    # No, we need to form a valid Sidon set of size n.
                    # But if we already have size n, we checked.
                    # If we have size < n, we can just add large numbers to avoid conflicts?
                    # If we add large numbers, we create new large diffs.
                    # We just need to ensure Sidon property.
                    pass
                
                last = s[-1]
                # Next element must be > last
                # Also, to minimize N, we should try smaller values?
                # But we are fixing N. So we try all valid next elements.
                
                # Optimization: 
                # If we add x, new diffs are {x - y for y in s}.
                # These must not be in diffs.
                # And x - y must be <= N.
                
                # Try candidates
                for x in range(last + 1, N + 1):
                    new_diffs = set()
                    valid = True
                    for y in s:
                        d = x - y
                        if d in diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid:
                        continue
                    
                    new_diffs_total = diffs.union(new_diffs)
                    new_covered = covered.union(target_diffs.intersection(new_diffs_total))
                    
                    # Prune: If we can't cover the remaining missing targets
                    # with the remaining slots?
                    # This is hard to estimate precisely, but:
                    # If len(new_covered) + (max possible new diffs) < 30, prune.
                    # Max possible new diffs if we add k more elements to current m elements:
                    # Each new element adds m + i diffs (where i is index of new element among new ones).
                    # Total new diffs from k elements added to m existing:
                    # Sum_{i=0}^{k-1} (m + i) = k*m + k(k-1)/2.
                    rem_slots = n - len(s) - 1
                    cur_size = len(s) + 1
                    max_possible_new_diffs = rem_slots * cur_size + rem_slots * (rem_slots - 1) // 2
                    # But some of these might be > 30 or duplicates.
                    # A loose bound: if len(new_covered) + max_possible_new_diffs < 30:
                    #    continue
                    
                    res = search(idx + 1, s + [x], new_diffs_total, new_covered)
                    if res is not None:
                        return res
                        
            res = search(0, s, diffs, covered)
            if res is not None:
                best_set = res
                break # Found for this N, try next N? No, we want min N.
                # So break N loop and return.
        
        if best_set is not None:
            break
            
    if best_set is None:
        # Fallback to greedy if search fails or times out
        a = [0]
        def diffs_list(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs_list(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                        a = trial
                        break
                    y += 1
        return a
        
    return best_set

