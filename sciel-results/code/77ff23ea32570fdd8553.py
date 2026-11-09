import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
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
        return all(d in diffs for d in range(1, 31))

    best_max = float('inf')
    best_set = None
    
    # Bitset-based backtracking solver
    # We search for Sidon sets of size k containing 0 that cover differences 1-30
    # Minimize the maximum element
    
    def solve_by_backtracking(max_size, time_limit):
        """Systematic backtracking with bitset tracking."""
        nonlocal best_max, best_set
        
        # State: (elements_tuple, used_diffs_set, covered_mask)
        # elements: sorted tuple starting with 0
        # used_diffs: set of all differences used (for Sidon check)
        # covered_mask: bitmask of which differences 1-30 are covered
        
        full_mask = (1 << 30) - 1
        target = set(range(1, 31))
        
        def backtrack(elements, used_diffs, covered_mask, depth):
            nonlocal best_max, best_set
            
            if time.time() - start_time > time_limit:
                return False
            
            # Check if we've covered all differences
            if covered_mask == full_mask:
                max_val = elements[-1] if elements else 0
                if max_val < best_max:
                    best_max = max_val
                    best_set = list(elements)
                    return True
                return False
            
            # If we've used max_size elements and haven't covered all, prune
            if len(elements) >= max_size:
                return False
            
            # Lower bound: we need at least ceil(30 / (len(elements)+1)) more elements
            # to cover remaining differences (each new element can cover at most len(elements)+1 new diffs)
            remaining = 30 - bin(covered_mask).count('1')
            needed = (remaining + len(elements)) // (len(elements) + 1)
            if len(elements) + needed > max_size:
                return False
            
            last = elements[-1] if elements else -1
            
            # Generate candidates: try small values first to minimize diameter
            # For each missing difference, try adding s+d for s in elements
            missing = [d for d in range(1, 31) if not (covered_mask & (1 << (d-1)))]
            
            # Sort missing diffs by priority: prefer diffs that are harder to cover
            # (larger diffs first, as they're more constrained)
            missing.sort(reverse=True)
            
            candidates = set()
            for d in missing:
                for s in elements:
                    x = s + d
                    if x > last and x <= 10**6:
                        candidates.add(x)
            
            # Also try some values near the current max to allow flexibility
            for offset in range(1, min(20, 10**6 - last)):
                candidates.add(last + offset)
            
            # Sort candidates: prefer smaller values (to keep diameter small)
            # but also prefer those that cover more missing diffs
            valid_candidates = []
            for x in candidates:
                if x <= last:
                    continue
                
                # Check Sidon property
                new_diffs = []
                valid = True
                for s in elements:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Compute new covered mask
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d-1))
                
                new_covered = bin(new_mask).count('1')
                new_missing = 30 - new_covered
                
                # Score: prefer smaller x, then more coverage
                valid_candidates.append((x, new_mask, new_diffs, new_missing))
            
            # Sort: primary by x (smaller better), secondary by new_missing (fewer better)
            valid_candidates.sort(key=lambda item: (item[0], item[3]))
            
            # Limit candidates to keep search manageable
            max_candidates = 50
            for x, new_mask, new_diffs, new_missing in valid_candidates[:max_candidates]:
                if time.time() - start_time > time_limit:
                    return False
                
                new_elements = elements + (x,)
                new_used = used_diffs.union(new_diffs)
                
                if backtrack(new_elements, new_used, new_mask, depth + 1):
                    return True
            
            return False
        
        # Start with element 0
        initial_elements = (0,)
        initial_used = set()
        initial_mask = 0
        
        return backtrack(initial_elements, initial_used, initial_mask, 0)
    
    # Try increasing sizes until we find a solution
    for size in range(8, 33):
        if time.time() - start_time > deadline:
            break
        
        # Use a time budget proportional to remaining time
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 5:
            break
        
        # Allocate more time for smaller sizes (easier to find solutions)
        time_budget = min(remaining_time * 0.8, 30)
        
        found = solve_by_backtracking(size, time.time() + time_budget)
        
        if found and best_set is not None:
            # Found a solution, now try to optimize with local search
            break
    
    # If no solution found, use fallback heuristic
    if best_set is None:
        # Simple greedy construction
        S = [0]
        covered = set()
        while not covered.issuperset(set(range(1, 31))) and len(S) <= 32:
            missing = set(range(1, 31)) - covered
            found = False
            for d in sorted(missing, reverse=True):
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
    
    # Local search polish on the found solution
    S = sorted(best_set[:])
    
    def try_reduce_element(S, i):
        """Try to reduce element at index i to any valid smaller value."""
        s_i = S[i]
        S_rest = S[:i] + S[i+1:]
        lower_bound = S[i-1] + 1 if i > 0 else 1
        
        for new_val in range(s_i - 1, lower_bound - 1, -1):
            if new_val in S_rest:
                continue
            S_new = S_rest + [new_val]
            S_new.sort()
            if is_sidon(S_new) and covers_all(S_new):
                return S_new
        return None
    
    # Iteratively reduce elements
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        S.sort()
        
        for i in range(len(S) - 1, 0, -1):
            if (time.time() - start_time) > deadline:
                break
            
            new_S = try_reduce_element(S, i)
            if new_S is not None:
                S = new_S
                improved = True
                break
    
    return S
