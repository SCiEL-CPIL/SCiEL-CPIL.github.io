import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    def is_valid_set(S):
        """Check if set S is valid."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i+1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add_small(S, a):
        """Check if a can be added to S where all elements of S > a."""
        # Need: for all b,c in S with b<=c, a ∤ (b+c)
        # b+c ranges from min(S)+min(S) to max(S)+max(S)
        # Check if any multiple of a is achievable as b+c
        min_sum = min(S) + min(S)
        max_sum = max(S) + max(S)
        # Find multiples of a in [min_sum, max_sum]
        first_mult = ((min_sum + a - 1) // a) * a
        for mult in range(first_mult, max_sum + 1, a):
            # Check if mult = b+c for some b,c in S with b<=c
            # b in [max(min(S), mult-max(S)), min(max(S), mult//2)]
            b_lo = max(min(S), mult - max(S))
            b_hi = min(max(S), mult // 2)
            if b_lo > b_hi:
                continue
            # Check if any b in [b_lo, b_hi] is in S and mult-b is in S
            # This is O(|S|) in worst case
            found = False
            for b in range(b_lo, b_hi + 1):
                if b in S and (mult - b) in S:
                    found = True
                    break
            if found:
                return False
        return True
    
    # Start with [171, 256]
    S = set(range(171, 257))
    
    # Try to add elements from 170 down to 1
    for a in range(170, 0, -1):
        if time.time() > deadline - 5.0:
            break
        if a not in S and can_add_small(S, a):
            S.add(a)
    
    S_list = sorted(S)
    
    # Local search: try to improve by swapping
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        S_sorted = sorted(S)
        for i in range(len(S_sorted)):
            if time.time() > deadline - 10.0:
                break
            removed = S_sorted[i]
            S_temp = set(S_sorted[:i] + S_sorted[i+1:])
            # Try to add two elements
            for a in range(170, 0, -1):
                if time.time() > deadline - 10.0:
                    break
                if a in S_temp:
                    continue
                # Check if a can be added
                min_S = min(S_temp)
                max_S = max(S_temp)
                if a < min_S:
                    if can_add_small(S_temp, a):
                        S_temp.add(a)
                        # Try to add one more
                        for b in range(a-1, 0, -1):
                            if time.time() > deadline - 10.0:
                                break
                            if b in S_temp:
                                continue
                            if can_add_small(S_temp, b):
                                S_temp.add(b)
                                break
                        if len(S_temp) > len(S):
                            S = S_temp
                            improved = True
                            break
                else:
                    # a >= min_S, need full check
                    S_test = sorted(S_temp | {a})
                    if is_valid_set(S_test):
                        S_temp.add(a)
                        for b in range(170, a+1, -1):
                            if time.time() > deadline - 10.0:
                                break
                            if b in S_temp:
                                continue
                            S_test2 = sorted(S_temp | {b})
                            if is_valid_set(S_test2):
                                S_temp.add(b)
                                break
                        if len(S_temp) > len(S):
                            S = S_temp
                            improved = True
                            break
            if improved:
                break
    
    return sorted(S)
