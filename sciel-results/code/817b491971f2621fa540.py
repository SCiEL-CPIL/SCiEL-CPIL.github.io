"""Rewritten program for Sidon set construction minimizing largest element."""

def construct():
    """
    Construct a Sidon set containing 0 with all positive differences distinct
    and every d in 1..30 occurring as a difference. Minimize the largest element.
    """
    required = set(range(1, 31))
    
    def get_all_diffs(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[j] - S[i])
        return diffs
    
    def is_sidon(S):
        seen = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    # Greedy construction: at each step, add the smallest x that
    # (1) maintains Sidon property, and
    # (2) covers at least one new required difference
    # Prefer x that covers more new required differences
    
    S = [0]
    current_diffs = set()
    
    max_search = 150
    
    for step in range(1, 32):
        remaining = required - current_diffs
        if not remaining:
            break
        
        best_x = None
        best_new_count = 0
        
        for x in range(1, max_search + 1):
            if x in S:
                continue
            
            # Check Sidon property
            new_diffs = set()
            valid = True
            for a in S:
                d = abs(x - a)
                if d > 0:
                    if d in current_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
            
            if not valid:
                continue
            
            # Count new required differences covered
            new_required = remaining & new_diffs
            num_new = len(new_required)
            
            if num_new > best_new_count:
                best_new_count = num_new
                best_x = x
        
        if best_x is None:
            max_search += 50
            continue
        
        S.append(best_x)
        S.sort()
        current_diffs = get_all_diffs(S)
    
    return S
