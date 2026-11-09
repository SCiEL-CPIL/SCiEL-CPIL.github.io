import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 80  # seconds, safety margin before 90s kill
    max_elements = 48
    n = 192
    
    S = []
    S_set = set()
    
    def is_safe_set(S_list):
        """Check if S_list is valid: no element divides sum of 3 distinct others."""
        for a in S_list:
            others = [v for v in S_list if v != a]
            L_len = len(others)
            if L_len < 3:
                continue
            
            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % a] += 1
            
            for i in range(L_len):
                u = others[i]
                for j in range(i + 1, L_len):
                    v = others[j]
                    target_rem = (-u - v) % a
                    cnt = rem_count.get(target_rem, 0)
                    if u % a == target_rem:
                        cnt -= 1
                    if v % a == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True
    
    # Pass 1: greedy from largest to smallest
    for x in range(n, 0, -1):
        if len(S) >= max_elements:
            break
        if time.time() - start_time > deadline - 5:
            break
        
        new_S = S + [x]
        if is_safe_set(new_S):
            S.append(x)
            S_set.add(x)
    
    # Pass 2: try to add remaining numbers (from small to large) that fit
    if len(S) < max_elements and time.time() - start_time < deadline - 10:
        for x in range(1, n + 1):
            if len(S) >= max_elements:
                break
            if time.time() - start_time > deadline - 5:
                break
            if x in S_set:
                continue
            new_S = S + [x]
            if is_safe_set(new_S):
                S.append(x)
                S_set.add(x)
    
    return S
