import time
import bisect

def construct():
    n = 256
    deadline = time.time() + 80
    
    def can_add(S_sorted, x):
        # S_sorted is sorted list, x not in S_sorted
        # Case 1: a = x, b > x, c >= b, b,c in S
        for i, b in enumerate(S_sorted):
            if b <= x:
                continue
            for k in range(i, len(S_sorted)):
                c = S_sorted[k]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: b = x, a < x, c >= x, a in S, c in S with c >= x
        # (x + c) % a != 0 for a in S, a < x, c in S, c >= x
        # Case 3: c = x, a < b <= x, a,b in S with b <= x
        # (b + x) % a != 0 for a,b in S, a < b <= x
        # Case 4: a = x, b = x, c > x, c in S
        # (x + c) % x = c % x != 0
        
        for a in S_sorted:
            if a >= x:
                break
            # Case 2: (x + c) % a for c in S, c >= x
            for c in S_sorted:
                if c >= x:
                    if (x + c) % a == 0:
                        return False
            # Case 3: (b + x) % a for b in S, a < b <= x
            for b in S_sorted:
                if a < b <= x:
                    if (b + x) % a == 0:
                        return False
            # Case 4: c % x != 0 for c in S, c > x
            for c in S_sorted:
                if c > x:
                    if c % x == 0:
                        return False
        
        return True
    
    best_size = 0
    best_set = None
    
    for start in range(120, 175):
        if time.time() > deadline:
            break
        S = sorted(range(start, n+1))
        
        for x in range(1, start):
            if time.time() > deadline:
                break
            if can_add(S, x):
                bisect.insort(S, x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    if best_set is None:
        return list(range(171, 257))
    return best_set
