"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _subset_sums_by_card(S):
    """Compute all subset sums grouped by cardinality using DP."""
    n = len(S)
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    for val in S:
        for k in range(n - 1, -1, -1):
            if sums_by_card[k]:
                for s in sums_by_card[k]:
                    sums_by_card[k + 1].add(s + val)
    return sums_by_card

def _is_valid_full(S):
    """Full exact validity check: no cross-cardinality sum collision."""
    n = len(S)
    if n == 0:
        return True
    sums_by_card = _subset_sums_by_card(S)
    for k1 in range(n + 1):
        for k2 in range(k1 + 1, n + 1):
            if sums_by_card[k1] & sums_by_card[k2]:
                return False
    return True

def _validity_check_minmax(S):
    """Quick sufficient condition: max_sum(k1) < min_sum(k2) for all k1 < k2."""
    n = len(S)
    if n == 0:
        return True
    S_sorted = sorted(S)
    for k1 in range(n + 1):
        for k2 in range(k1 + 1, n + 1):
            min_sum_k2 = sum(S_sorted[:k2]) if k2 > 0 else 0
            max_sum_k1 = sum(S_sorted[-k1:]) if k1 > 0 else 0
            if max_sum_k1 >= min_sum_k2:
                return False
    return True

def _can_add(S, val):
    """Check if adding val to S preserves the cardinality-separated property.
    Uses incremental DP: only new sums involving val need checking against existing layers."""
    n = len(S)
    if n == 0:
        return True
    
    # Quick min/max pre-check
    S_new_sorted = sorted(S + [val])
    n_new = n + 1
    for k1 in range(n_new + 1):
        for k2 in range(k1 + 1, n_new + 1):
            min_sum_k2 = sum(S_new_sorted[:k2]) if k2 > 0 else 0
            max_sum_k1 = sum(S_new_sorted[-k1:]) if k1 > 0 else 0
            if max_sum_k1 >= min_sum_k2:
                return False
    
    # Exact check: compute sums by cardinality for S, then check new sums
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    for v in S:
        for k in range(n - 1, -1, -1):
            if sums_by_card[k]:
                for s in sums_by_card[k]:
                    sums_by_card[k + 1].add(s + val if False else s + v)
    
    # New sums: for each k, new_sum_k = {s + val for s in sums_by_card[k-1]}
    # Check: new sums at cardinality k must not collide with any existing sum at cardinality j != k
    # Also new sums at cardinality k must not collide with new sums at cardinality k' != k (but they can't since different cardinalities)
    # Actually we just need: for each new sum at level k, it's not in any sums_by_card[j] for j != k
    # And also not equal to another new sum at a different level (impossible since different levels)
    
    for k in range(1, n + 1):
        if not sums_by_card[k - 1]:
            continue
        for s in sums_by_card[k - 1]:
            new_s = s + val
            # Check against all existing layers except k (which we're building)
            for j in range(n + 1):
                if j == k:
                    continue
                if new_s in sums_by_card[j]:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_set = []
    best_size = 0
    
    # Backtracking search: try adding elements from large to small
    # This tends to separate sum layers more effectively
    current = []
    current_set = set()
    
    def backtrack(remaining_candidates, depth):
        nonlocal best_set, best_size, current, current_set
        
        if time.time() > deadline:
            return
        
        if len(current) > best_size:
            best_set = list(current)
            best_size = len(current)
        
        if not remaining_candidates:
            return
        
        # Try each candidate in descending order (larger values separate layers better)
        for i, val in enumerate(remaining_candidates):
            if time.time() > deadline:
                return
            
            # Quick min/max check before expensive exact check
            test_set = sorted(current + [val])
            n_test = len(test_set)
            valid = True
            for k1 in range(n_test + 1):
                for k2 in range(k1 + 1, n_test + 1):
                    min_sum_k2 = sum(test_set[:k2]) if k2 > 0 else 0
                    max_sum_k1 = sum(test_set[-k1:]) if k1 > 0 else 0
                    if max_sum_k1 >= min_sum_k2:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Exact check
            if _can_add(current, val):
                current.append(val)
                current_set.add(val)
                backtrack(remaining_candidates[i+1:], depth + 1)
                current.pop()
                current_set.discard(val)
    
    # Start with descending order of all candidates 256..1
    candidates = list(range(256, 0, -1))
    backtrack(candidates, 0)
    
    # If we have time, try a second pass with a different ordering
    if time.time() < deadline - 5 and best_size < 60:
        # Try ascending order starting from small values
        current = []
        current_set = set()
        
        def backtrack_asc(remaining_candidates, depth):
            nonlocal best_set, best_size, current, current_set
            
            if time.time() > deadline:
                return
            
            if len(current) > best_size:
                best_set = list(current)
                best_size = len(current)
            
            if not remaining_candidates:
                return
            
            for i, val in enumerate(remaining_candidates):
                if time.time() > deadline:
                    return
                
                test_set = sorted(current + [val])
                n_test = len(test_set)
                valid = True
                for k1 in range(n_test + 1):
                    for k2 in range(k1 + 1, n_test + 1):
                        min_sum_k2 = sum(test_set[:k2]) if k2 > 0 else 0
                        max_sum_k1 = sum(test_set[-k1:]) if k1 > 0 else 0
                        if max_sum_k1 >= min_sum_k2:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                if _can_add(current, val):
                    current.append(val)
                    current_set.add(val)
                    backtrack_asc(remaining_candidates[i+1:], depth + 1)
                    current.pop()
                    current_set.discard(val)
        
        candidates_asc = list(range(1, 257))
        backtrack_asc(candidates_asc, 0)
    
    # Try a third strategy: greedy with minimum-degree heuristic
    if time.time() < deadline - 10 and best_size < 60:
        current = []
        current_set = set()
        
        # Build a conflict graph: two values conflict if they can't coexist in any valid set
        # Approximation: check pairwise if {a, b} is valid
        # For speed, use min/max check on {a, b}
        
        # Greedy: repeatedly pick the value that has fewest conflicts with remaining candidates
        remaining = list(range(256, 0, -1))
        
        while remaining and time.time() < deadline:
            # For each remaining candidate, count how many other remaining candidates it conflicts with
            best_candidate = None
            best_conflicts = float('inf')
            
            for i, val in enumerate(remaining):
                if time.time() > deadline:
                    break
                
                conflicts = 0
                for j, other in enumerate(remaining):
                    if i == j:
                        continue
                    # Check if val and other can coexist with current set
                    test_set = sorted(current + [val, other])
                    n_test = len(test_set)
                    valid = True
                    for k1 in range(n_test + 1):
                        for k2 in range(k1 + 1, n_test + 1):
                            min_sum_k2 = sum(test_set[:k2]) if k2 > 0 else 0
                            max_sum_k1 = sum(test_set[-k1:]) if k1 > 0 else 0
                            if max_sum_k1 >= min_sum_k2:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        conflicts += 1
                
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_candidate = val
            
            if best_candidate is None:
                break
            
            # Verify best_candidate can actually be added
            test_set = sorted(current + [best_candidate])
            n_test = len(test_set)
            valid = True
            for k1 in range(n_test + 1):
                for k2 in range(k1 + 1, n_test + 1):
                    min_sum_k2 = sum(test_set[:k2]) if k2 > 0 else 0
                    max_sum_k1 = sum(test_set[-k1:]) if k1 > 0 else 0
                    if max_sum_k1 >= min_sum_k2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and _can_add(current, best_candidate):
                current.append(best_candidate)
                current_set.add(best_candidate)
                remaining.remove(best_candidate)
                if len(current) > best_size:
                    best_set = list(current)
                    best_size = len(current)
            else:
                remaining.remove(best_candidate)
    
    # Final: ensure valid result
    if best_size > 0:
        # Deduplicate and range-check
        seen = set()
        cleaned = []
        for x in best_set:
            if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 256 and x not in seen:
                seen.add(x)
                cleaned.append(x)
        best_set = cleaned
        
        # Exact final verification for the returned set
        if len(best_set) <= 20:
            if not _is_valid_full(best_set):
                # Fallback
                best_set = list(range(1, 9))
        else:
            # For larger sets, trust the min/max check which is sufficient
            if not _validity_check_minmax(best_set):
                # Try to trim until valid
                while best_set and not _validity_check_minmax(best_set):
                    best_set.pop()
                if not best_set:
                    best_set = list(range(1, 9))
    
    return list(best_set)
# EVOLVE-BLOCK-END
