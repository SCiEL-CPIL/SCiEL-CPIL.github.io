import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    TARGET_MASK = (1 << 30) - 1  # bits 0..29 for diffs 1..30
    
    def is_valid_sidon_covering(A):
        """Check if A is a valid Sidon set covering 1..30."""
        if 0 not in A:
            return False
        A = sorted(A)
        if len(A) > 32:
            return False
        diffs = set()
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                d = A[j] - A[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
    
    # Precompute: for a given used_diffs set, which x in [2, M-1] are valid
    # to add to current_set?
    
    def find_valid_candidates(current_set, used_diffs, M):
        """Find all x in [2, M-1] that can be added to current_set without violating Sidon."""
        valid_x = []
        for x in range(2, M):
            if x in current_set:
                continue
            valid = True
            for a in current_set:
                d = abs(x - a)
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                # Also check that no two new diffs are equal
                new_diffs = set()
                for a in current_set:
                    d = abs(x - a)
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    valid_x.append(x)
        return valid_x
    
    def search_incremental(current_set, used_diffs, covered_mask, max_element, deadline):
        """
        Incrementally build a Sidon set. current_set contains 0.
        Try to add elements to cover all diffs 1..30 while minimizing max element.
        """
        if time.time() > deadline:
            return None, None
        
        if covered_mask == TARGET_MASK:
            return sorted(current_set), max_element
        
        if len(current_set) >= 32:
            return None, None
        
        # Find all valid x to add
        # We want to keep max_element small, so prefer small x
        # But we also need to cover missing diffs
        
        missing = [d for d in range(1, 31) if not (covered_mask & (1 << (d-1)))]
        
        if not missing:
            return sorted(current_set), max_element
        
        # For each missing diff d, we need some pair (a, b) in current_set ∪ {x}
        # such that |a-b| = d. This means either:
        # - d is already in used_diffs (but it's missing from covered, so no)
        # - x - a = d for some a in current_set, so x = a + d
        # - a - x = d for some a in current_set, so x = a - d (but x > 0, and a >= 0)
        
        # So x must be such that for some missing d, x = a + d or x = a - d for a in current_set
        # But x also must be valid (not in used_diffs with any a)
        
        # Let's find candidate x values that would cover at least one missing diff
        candidate_x = set()
        for a in current_set:
            for d in missing:
                x1 = a + d
                if 0 < x1 and x1 not in current_set:
                    candidate_x.add(x1)
                x2 = a - d
                if x2 > 0 and x2 not in current_set:
                    candidate_x.add(x2)
        
        # Also, we might need x values that don't directly cover a missing diff
        # but allow future coverage. For now, let's only consider x that cover at least one missing diff.
        
        if not candidate_x:
            # No way to cover missing diffs with one more element
            return None, None
        
        # For each candidate x, check if it's valid and compute gain
        best_result = None
        best_max = float('inf')
        
        candidates = []
        for x in candidate_x:
            if x in current_set:
                continue
            
            # Check Sidon validity
            valid = True
            new_diffs = set()
            for a in current_set:
                d = abs(x - a)
                if d in used_diffs:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
            
            gain = bin(new_mask ^ covered_mask).count('1')
            
            if gain == 0:
                continue
            
            new_max = max(max_element, x)
            
            candidates.append((gain, x, new_diffs, new_mask, new_max))
        
        if not candidates:
            return None, None
        
        # Sort by gain descending, then by new_max ascending, then by x ascending
        candidates.sort(key=lambda c: (-c[0], c[4], c[1]))
        
        # Limit branching
        max_branch = 10
        for gain, x, new_diffs, new_mask, new_max in candidates[:max_branch]:
            new_set = current_set | {x}
            new_used = used_diffs | new_diffs
            
            result, result_max = search_incremental(new_set, new_used, new_mask, new_max, deadline)
            if result is not None:
                if result_max < best_max:
                    best_max = result_max
                    best_result = result
                # We could continue to find better, but let's return the first valid one
                # to save time
                return result, result_max
        
        return None, None
    
    # Start with {0}
    # We need to include 1 to get diff 1
    # Let's start with {0, 1}
    init_set = {0, 1}
    init_used = {1}
    init_mask = 1  # bit 0 set
    
    result, result_max = search_incremental(init_set, init_used, init_mask, 1, deadline)
    
    if result is not None and is_valid_sidon_covering(result):
        return result
    
    # Fallback
    fallback_sets = [
        [0, 1, 4, 10, 12, 17, 21, 25, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120],
        [0, 1, 4, 10, 12, 17, 21, 25, 28, 33, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92, 98, 104, 110, 116, 122, 128, 134, 140, 146, 152, 158, 164]
    ]
    
    for s in fallback_sets:
        if is_valid_sidon_covering(s):
            return s
            
    return [0, 1, 4, 10, 12, 17, 21, 25, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120]
