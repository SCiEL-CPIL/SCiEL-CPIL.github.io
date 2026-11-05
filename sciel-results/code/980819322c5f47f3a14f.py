import time

def construct():
    n = 256
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # Start with the known valid large set
    k = 171
    S = set(range(k, n + 1))
    
    def is_valid_with(S, a):
        """Check if adding a to S maintains validity."""
        S_list = sorted(S)
        larger = [x for x in S_list if x > a]
        for i, b in enumerate(larger):
            for c in larger[i:]:
                if (b + c) % a == 0:
                    return False
        return True
    
    # Try to add numbers from 1 upward
    for a in range(1, k):
        if time.time() > deadline:
            break
        if is_valid_with(S, a):
            S.add(a)
    
    # Try descending order on a fresh copy
    S2 = set(range(k, n + 1))
    for a in range(k - 1, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_with(S2, a):
            S2.add(a)
    
    # Try ascending again on S2
    for a in range(1, k):
        if time.time() > deadline:
            break
        if is_valid_with(S2, a):
            S2.add(a)
    
    # Pick the larger set
    if len(S2) > len(S):
        S = S2
    
    # Final verification
    S_list = sorted(S)
    valid = True
    for i, a in enumerate(S_list):
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            for c in S_list[j:]:
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return list(range(171, 257))
    
    return S_list
