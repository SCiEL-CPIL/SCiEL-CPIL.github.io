import math
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 85
    
    def check_deadline():
        return (time.time() - start_time) > TIME_LIMIT

    N = 512
    odds = [x for x in range(1, N + 1, 2)]
    
    # Precompute LCM for odd pairs
    # lcms[i][j] for odds[i], odds[j]
    # To save memory and time, we can just compute on the fly in the clique search
    # or precompute a boolean matrix compat_odd
    num_odds = len(odds)
    compat_odd = [[False] * num_odds for _ in range(num_odds)]
    
    for i in range(num_odds):
        a = odds[i]
        for j in range(i, num_odds):
            b = odds[j]
            g = math.gcd(a, b)
            l = a // g * b
            if l <= N:
                compat_odd[i][j] = True
                compat_odd[j][i] = True
            else:
                compat_odd[i][j] = False
                compat_odd[j][i] = False

    best_set = set([1])
    best_score = 0
    
    # Strategy 1: Greedy Clique Building
    # We want to maximize |C| + floor(log2(512/max(C))) + 1
    # Let's try to build cliques greedily.
    # Sort odds by value, or by degree?
    # Smaller values allow more powers of 2.
    # So we should prefer smaller values.
    
    # Try greedy addition in increasing order
    C = set([1])
    for x in odds:
        if check_deadline():
            break
        if x in C:
            continue
        # Check if x is compatible with all in C
        ok = True
        for y in C:
            idx_x = odds.index(x) # This is slow. Use a map.
            idx_y = odds.index(y)
            if not compat_odd[idx_x][idx_y]:
                ok = False
                break
        if ok:
            C.add(x)
    
    # Calculate score for this C
    max_c = max(C)
    k = int(math.log2(N // max_c)) if max_c > 0 else 9
    # Ensure 2^k * max_c <= N
    while (1 << k) * max_c > N:
        k -= 1
    score = len(C) + k + 1
    if score > best_score:
        best_score = score
        best_set = set(C)
    
    # Strategy 2: Local Search on the Clique
    # Try to swap elements to increase score
    # Score depends on |C| and max(C).
    # Removing a large element might allow adding two smaller ones?
    # Or removing a large element reduces max(C), increasing k, which might compensate for loss of 1 in |C|?
    # If max(C) drops from 300 to 100, k goes from 0 to 2. Gain 2 in k. Loss 1 in |C|. Net +1.
    
    def get_score(C):
        if not C:
            return 0
        max_c = max(C)
        k = int(math.log2(N // max_c)) if max_c > 0 else 9
        while (1 << k) * max_c > N:
            k -= 1
        return len(C) + k + 1

    # Convert C to a list for easier manipulation
    current_C = list(best_set)
    improved = True
    while improved and not check_deadline():
        improved = False
        # Try removing each element
        for i in range(len(current_C)):
            if check_deadline():
                break
            u = current_C[i]
            new_C = set(current_C)
            new_C.discard(u)
            
            # Try to add any other odd number
            added = False
            for x in odds:
                if x in new_C:
                    continue
                # Check compatibility
                idx_x = odds.index(x)
                ok = True
                for y in new_C:
                    idx_y = odds.index(y)
                    if not compat_odd[idx_x][idx_y]:
                        ok = False
                        break
                if ok:
                    new_C.add(x)
                    added = True
                    break
            
            if get_score(new_C) > get_score(set(current_C)):
                current_C = list(new_C)
                improved = True
                break
    
    if get_score(set(current_C)) > best_score:
        best_score = get_score(set(current_C))
        best_set = set(current_C)
        
    # Strategy 3: Try specific highly composite odd divisors
    # We already did this implicitly, but let's be explicit for top candidates
    candidates = [1, 3, 5, 7, 9, 15, 21, 25, 27, 35, 45, 63, 75, 105, 135, 225, 315, 375, 405, 441]
    for D in candidates:
        if check_deadline():
            break
        C = set()
        for d in range(1, D + 1, 2):
            if D % d == 0:
                C.add(d)
        # Verify it's a clique (it should be)
        # Calculate score
        s = get_score(C)
        if s > best_score:
            best_score = s
            best_set = set(C)
            
    # Construct final set
    max_c = max(best_set) if best_set else 1
    k = int(math.log2(N // max_c)) if max_c > 0 else 9
    while (1 << k) * max_c > N:
        k -= 1
        
    powers = [1 << i for i in range(k + 1)]
    final_set = list(best_set) + powers
    
    # Remove duplicates (1 is in both)
    final_set = list(set(final_set))
    
    # Verification
    valid = True
    for i in range(len(final_set)):
        for j in range(i + 1, len(final_set)):
            a, b = final_set[i], final_set[j]
            if math.lcm(a, b) > N:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return [1]
        
    return final_set
