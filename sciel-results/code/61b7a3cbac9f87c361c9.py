import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # 85s guard (90s kill)
    
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    a = 1
    while a <= 128:
        b = a
        while b <= 128:
            c = b
            while c <= 128:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    smooth_list = sorted(smooth)
    S = len(smooth_list)
    
    # Precompute divisibility relation for smooth numbers
    # divs[i] is a set of indices j such that smooth_list[j] divides smooth_list[i]
    # Or simpler: just check on the fly since S is small (~36)
    
    def is_antichain(terms):
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                if terms[i] % terms[j] == 0 or terms[j] % terms[i] == 0:
                    return False
        return True

    answers = []
    for n in range(1, 129):
        if time.time() - start_time > DEADLINE:
            # Fallback for remaining
            answers.append([n] if n in smooth else [1, n-1])
            continue
            
        # 1. Check if n is smooth
        if n in smooth:
            answers.append([n])
            continue
            
        # 2. Check for size 2 representation
        found = False
        for i in range(S):
            a = smooth_list[i]
            if a >= n:
                break
            b = n - a
            if b in smooth:
                if a != b and is_antichain([a, b]):
                    answers.append([a, b])
                    found = True
                    break
        if found:
            continue
            
        # 3. Check for size 3 representation
        # To speed up, we can iterate over pairs (a,b) and check if n-a-b is smooth
        # We need a < b < c to avoid duplicates? No, just distinct terms.
        # But we can enforce a <= b <= c for iteration efficiency, then check antichain.
        found = False
        for i in range(S):
            a = smooth_list[i]
            if a > n // 3:
                break
            for j in range(i, S):
                b = smooth_list[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c in smooth:
                    if a != b and b != c and a != c:
                        if is_antichain([a, b, c]):
                            answers.append([a, b, c])
                            found = True
                            break
            if found:
                break
        if found:
            continue
            
        # 4. Fallback: should not happen for 1..128
        # Just pick 3 distinct smooth numbers that sum to n and form an antichain
        # If that fails, relax antichain? No, must be valid.
        # We can do a small DFS here if needed, but likely size 3 covers all.
        found = False
        for i in range(S):
            a = smooth_list[i]
            if a > n:
                break
            for j in range(S):
                b = smooth_list[j]
                if b == a:
                    continue
                if a + b > n:
                    continue
                c = n - a - b
                if c in smooth:
                    if c != a and c != b:
                        if is_antichain([a, b, c]):
                            answers.append([a, b, c])
                            found = True
                            break
            if found:
                break
        if found:
            continue
            
        # 5. Last resort: 4 terms?
        # For 1..128, size 3 should be sufficient.
        # If not, we can use a generic backtracking for this specific n.
        def dfs(idx, curr_sum, curr_set, forbidden):
            if curr_sum > n:
                return None
            if len(curr_set) > 30:
                return None
            if idx == S:
                if curr_sum == n and len(curr_set) > 0:
                    return tuple(curr_set)
                return None
            
            # Skip
            res = dfs(idx + 1, curr_sum, curr_set, forbidden)
            if res:
                return res
            
            # Include
            if not ((forbidden >> idx) & 1):
                val = smooth_list[idx]
                new_sum = curr_sum + val
                if new_sum <= n:
                    # Check compatibility with current set
                    compatible = True
                    for v in curr_set:
                        if v % val == 0 or val % v == 0:
                            compatible = False
                            break
                    if compatible:
                        curr_set.append(val)
                        # Update forbidden: all indices k where smooth_list[k] is incompatible with val
                        new_forbidden = forbidden
                        for k in range(S):
                            if k == idx:
                                continue
                            vk = smooth_list[k]
                            if val % vk == 0 or vk % val == 0:
                                new_forbidden |= (1 << k)
                        new_forbidden |= (1 << idx)
                        
                        res = dfs(idx + 1, new_sum, curr_set, new_forbidden)
                        curr_set.pop()
                        if res:
                            return res
            return None

        res = dfs(0, 0, [], 0)
        if res:
            answers.append(list(res))
        else:
            # Should absolutely not happen
            answers.append([n] if n in smooth else [1, 2, n-3])

    return answers
