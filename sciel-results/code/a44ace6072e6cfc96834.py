import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    best_len = 0
    best_start = 1
    
    # Two-pointer / Sliding window approach
    # For each start s, find the maximum end e such that [s, e] is valid.
    # We can optimize by noting that as s increases, the valid end e can only increase or stay same?
    # Actually, if [s, e] is valid, [s+1, e] is not necessarily valid because removing s might not help if the conflict was with s?
    # No, removing an element only reduces conflicts. So if [s, e] is valid, any sub-interval is valid.
    # But the property "no x, y in interval are inverses" is monotonic with respect to interval inclusion?
    # Yes, if [s, e] has no inverse pairs, then [s', e'] subset [s, e] has no inverse pairs.
    # However, extending the interval can introduce new pairs.
    # The standard O(N^2) approach is to fix s and increase e until a conflict is found.
    # Can we do O(N)?
    # Let e be the max end for start s.
    # When we move to s+1, the new max end e' must be >= e.
    # Why? Because [s+1, e] is a sub-interval of [s, e], so it is valid.
    # We can try to extend e further.
    # So we can use a single pointer e that only moves forward.
    
    e = 0
    in_window = [False] * (N + 1)
    # But we need to remove elements from the window as s increases.
    # So we need to track counts or just use a set.
    
    window_elements = set()
    e = 0
    for s in range(1, N + 1):
        # Remove s-1 from window? No, s is the new start.
        # The window was [s-1, e_prev]. Now we want [s, e_new].
        # First, remove s-1 from the window structure.
        if s > 1:
            window_elements.discard(s-1)
            
        # Now try to extend e as far as possible.
        # Current window is [s, e] (if e >= s).
        # If e < s, we start with empty window and e = s-1.
        if e < s - 1:
            e = s - 1
            
        while e < N:
            candidate = e + 1
            # Check if candidate can be added to [s, e]
            # 1. candidate itself must not be self-inverse
            if inv[candidate] == candidate:
                break
            # 2. candidate's inverse must not be in the current window [s, e]
            # Note: if candidate's inverse is candidate, we already broke.
            # So inv[candidate] != candidate.
            if inv[candidate] in window_elements:
                break
            
            # Check if candidate's inverse is candidate? Handled above.
            # Is it possible that adding candidate creates a conflict with a future element?
            # No, we only care about the current interval.
            
            # Safe to add
            window_elements.add(candidate)
            e = candidate
            
        current_len = e - s + 1
        if current_len > best_len:
            best_len = current_len
            best_start = s

    if best_len == 0:
        # Fallback: find any valid single element
        for x in range(1, N + 1):
            if inv[x] != x:
                best_start = x
                best_len = 1
                break

    return [best_start, best_len]
