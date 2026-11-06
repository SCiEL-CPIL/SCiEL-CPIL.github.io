"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def _is_valid(a_set, n=256):
    """Check if the set satisfies the property: for all a<b<=c in set, a does not divide b+c."""
    s = sorted(a_set)
    s_set = set(s)
    for i, a in enumerate(s):
        for j in range(i+1, len(s)):
            b = s[j]
            for k in range(j, len(s)):
                c = s[k]
                if (b + c) % a == 0:
                    return False
    return True

def _check_validity(a_set):
    """Faster validity check using precomputed conflicts."""
    s = sorted(a_set)
    s_set = set(s)
    for i, a in enumerate(s):
        # For this a, check all pairs (b,c) where b > a, c >= b, both in set
        # b+c must not be divisible by a
        for j in range(i+1, len(s)):
            b = s[j]
            # Find smallest multiple of a that is >= b + a + 1 (since c >= b > a)
            # Actually c >= b, so b+c >= 2b
            # We need b+c to be a multiple of a
            # b+c = m*a for some integer m
            # c = m*a - b
            # c must be in [b, max(s)] and in s_set
            for m in range((b + b + a - 1) // a, (b + s[-1] + a - 1) // a + 1):
                c_val = m * a - b
                if c_val < b:
                    continue
                if c_val > s[-1]:
                    break
                if c_val in s_set:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second safety margin
    
    n = 256
    
    # Start with the known good interval [171, 256]
    current = set(range(171, 257))
    best = set(current)
    best_size = len(current)
    
    # Try to add elements from below
    for candidate in range(1, 171):
        if time.time() > deadline - 2:
            break
        
        test_set = current | {candidate}
        if _check_validity(test_set):
            current = test_set
            best = set(current)
            best_size = len(current)
    
    # Now try to swap: remove one element, add another
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for to_remove in sorted(current):
            if time.time() > deadline - 3:
                break
            remaining = current - {to_remove}
            for to_add in range(1, n+1):
                if to_add in current:
                    continue
                if time.time() > deadline - 3:
                    break
                test_set = remaining | {to_add}
                if len(test_set) > best_size and _check_validity(test_set):
                    current = test_set
                    best = set(current)
                    best_size = len(current)
                    changed = True
                    break
    
    # Try to add more elements after swaps
    for candidate in range(1, n+1):
        if time.time() > deadline - 2:
            break
        if candidate in current:
            continue
        test_set = current | {candidate}
        if len(test_set) > best_size and _check_validity(test_set):
            current = test_set
            best = set(current)
            best_size = len(current)
    
    # Final check
    if not _check_validity(best):
        # Fallback to safe interval
        return list(range(171, 257))
    
    return sorted(best)
# EVOLVE-BLOCK-END
