import time
import bisect

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 256
    
    def is_valid_addition_fast(a, S_set):
        """Check if a can be added. All elements in S are > a."""
        # b, c in S, b > a, c >= b.
        # Min sum: 2(a+1). Max sum: 2n.
        # Multiples of a in [2(a+1), 2n]
        
        # k*a >= 2a+2 => k >= 2 + 2/a
        # For a >= 2, k_min = 3. For a=1, k_min = 4.
        k_min = 2
        while k_min * a < 2 * (a + 1):
            k_min += 1
            
        k_max = 2 * n // a
        
        for k in range(k_min, k_max + 1):
            M = k * a
            # We need b, c in S such that b + c = M, a < b <= c
            # b must be at least a+1 and at most M//2
            # c = M - b must be in S and c >= b
            # b in [max(a+1, M-n), M//2]
            
            b_start = max(a + 1, M - n)
            b_end = M // 2
            
            for b in range(b_start, b_end + 1):
                if b in S_set and (M - b) in S_set:
                    return False
        return True

    def full_validate(S_list):
        n2 = len(S_list)
        S_set = set(S_list)
        for i in range(n2):
            a = S_list[i]
            for j in range(i+1, n2):
                b = S_list[j]
                for k in range(j, n2):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Start with the known valid interval [171, 256]
    S_list = list(range(171, n+1))
    S_set = set(S_list)
    
    # Greedy addition in descending order
    for a in range(170, 0, -1):
        if time.time() > deadline - 5:
            break
        if is_valid_addition_fast(a, S_set):
            pos = bisect.bisect_left(S_list, a)
            S_list.insert(pos, a)
            S_set.add(a)
            
    # Try to add any remaining elements in a second pass (ascending) just in case
    # (Though descending should be sufficient for this constraint structure)
    for a in range(1, n+1):
        if time.time() > deadline - 2:
            break
        if a not in S_set:
            if is_valid_addition_fast(a, S_set):
                pos = bisect.bisect_left(S_list, a)
                S_list.insert(pos, a)
                S_set.add(a)
                
    # Final validation
    if time.time() < deadline:
        if len(S_list) > 5 and not full_validate(S_list):
            return list(range(171, n+1))
            
    return S_list
