"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _has_4ap(s, x):
    """Check if adding x to set s creates a 4-term AP."""
    s_set = set(s)
    # Case 1: x is the last term: a, a+d, a+2d, x
    # x - 3d = a, x - 2d = a+d, x - d = a+2d
    for d in range(1, (x-1)//3 + 1):
        if (x - d) in s_set and (x - 2*d) in s_set and (x - 3*d) in s_set:
            return True
    
    # Case 2: x is the third term: a, a+d, x, x+d
    # a = x - 2d, a+d = x - d, x, x+d
    # We need x-d in s, x-2d in s, and x+d in s? No, x+d is not in s yet.
    # Wait, the AP is a, a+d, a+2d, a+3d.
    # If x = a+2d, then a = x-2d, a+d = x-d, a+3d = x+d.
    # So we need x-2d in s, x-d in s, and x+d in s.
    # But x+d > x, so x+d cannot be in s (since we add in increasing order).
    # So this case is impossible if we add in increasing order.
    
    # Case 3: x is the second term: a, x, x+d, x+2d
    # a = x-d, x, x+d, x+2d
    # Need x-d in s, x+d in s, x+2d in s.
    # x+d and x+2d are > x, so they cannot be in s.
    # Impossible.
    
    # Case 4: x is the first term: x, x+d, x+2d, x+3d
    # Need x+d, x+2d, x+3d in s.
    # All > x, so impossible.
    
    return False

def construct():
    deadline = time.time() + 50.0  # Safety margin
    selected = []
    selected_set = set()
    
    for x in range(1, 513):
        if time.time() > deadline:
            break
        
        # Check if adding x creates a 4-term AP
        # Only need to check if x is the 4th term (largest), since all other terms
        # in any AP containing x would have to be smaller than x (which are already
        # in selected_set) or larger (which can't be, since we're building in order).
        # Actually, let's re-verify.
        # An AP is a, a+d, a+2d, a+3d.
        # If x is in the AP, x can be a, a+d, a+2d, or a+3d.
        # If x = a, then a+d, a+2d, a+3d must be in the final set. But they are > x,
        # so they haven't been added yet. We can't check this now.
        # However, if we add x now, we might later add a+d, a+2d, a+3d, creating an AP.
        # The greedy algorithm only checks if the *current* set has an AP.
        # So we only need to check if adding x creates an AP *within the current set plus x*.
        # That means all 4 terms must be in {selected} ∪ {x}.
        # Since x is the largest element being considered, the only way x can be part of
        # a 4-term AP with 3 existing elements is if x is the largest term (a+3d).
        # If x were a, a+d, or a+2d, the other terms would be larger than x, and thus
        # not in selected yet. So they can't form an AP with existing elements.
        
        # So we only need to check:
        # Does there exist d > 0 such that x-d, x-2d, x-3d are all in selected_set?
        
        is_safe = True
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - d) in selected_set and (x - 2*d) in selected_set and (x - 3*d) in selected_set:
                is_safe = False
                break
        
        if is_safe:
            selected.append(x)
            selected_set.add(x)
    
    return selected
# EVOLVE-BLOCK-END
