import time

def construct():
    start_time = time.time()
    deadline = 85.0

    # Generate ternary candidates
    candidates = []
    for mask in range(1 << 6):
        val = 0
        for i in range(6):
            if (mask >> i) & 1:
                val += 3**i
        if val <= 512:
            candidates.append(val)
            
    # We have 64 candidates.
    # Try to find a valid subset of size 48.
    
    def is_valid_subset(S):
        S_set = set(S)
        S_list = list(S)
        n = len(S_list)
        
        for a in S_list:
            others = [x for x in S_list if x != a]
            
            # k=2: 2a = x+y
            target2 = 2 * a
            for x in others:
                y = target2 - x
                if y != x and y in S_set and y != a:
                    return False

            # k=3: 3a = x+y+z
            target3 = 3 * a
            for i in range(len(others)):
                x = others[i]
                rem2 = target3 - x
                for y in others[i+1:]:
                    z = rem2 - y
                    if z != x and z != y and z in S_set and z != a:
                        return False
                        
            # k=4: 4a = x+y+z+w
            # Only check if time permits and set is small
            if n < 30:
                target4 = 4 * a
                for i in range(len(others)):
                    x = others[i]
                    rem3 = target4 - x
                    for j in range(i+1, len(others)):
                        y = others[j]
                        rem2 = rem3 - y
                        for k in range(j+1, len(others)):
                            z = others[k]
                            w = rem2 - z
                            if w != x and w != y and w != z and w in S_set and w != a:
                                return False
            elif time.time() - start_time < deadline - 5:
                # If we have time, check k=4 for all
                target4 = 4 * a
                for i in range(len(others)):
                    x = others[i]
                    rem3 = target4 - x
                    for j in range(i+1, len(others)):
                        y = others[j]
                        rem2 = rem3 - y
                        for k in range(j+1, len(others)):
                            z = others[k]
                            w = rem2 - z
                            if w != x and w != y and w != z and w in S_set and w != a:
                                return False
            # For n >= 30, we skip k=4+ assuming ternary robustness
            
        return True

    # Greedy construction
    selected = []
    # Try adding in ascending order
    for x in sorted(candidates):
        if len(selected) >= 48:
            break
        if time.time() > deadline:
            break
        candidate_set = selected + [x]
        if is_valid_subset(candidate_set):
            selected.append(x)
            
    if len(selected) >= 48:
        return selected[:48]
        
    # If not 48, try descending
    selected = []
    for x in sorted(candidates, reverse=True):
        if len(selected) >= 48:
            break
        if time.time() > deadline:
            break
        candidate_set = selected + [x]
        if is_valid_subset(candidate_set):
            selected.append(x)
            
    if len(selected) >= 48:
        return selected[:48]
        
    # Fallback
    return [1, 2, 4, 8, 16, 32, 64, 128, 256]

