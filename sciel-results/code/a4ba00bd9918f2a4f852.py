import time
import random

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    # 1. Generate base-3 set (digits 0,1 only)
    base3_set = set()
    for x in range(1, n + 1):
        # Check if base-3 representation has only 0s and 1s
        y = x
        valid = True
        while y > 0:
            if y % 3 == 2:
                valid = False
                break
            y //= 3
        if valid:
            base3_set.add(x)
            
    best_set = set(base3_set)
    best_size = len(best_set)
    
    def can_add(y, S):
        # Check if adding y to S creates a 4-AP
        # y is part of a 4-AP with 3 elements in S
        # Cases: y is t0, t1, t2, or t3
        
        # y = t0: y, y+d, y+2d, y+3d. Need y+d, y+2d, y+3d in S
        # Max d: y+3d <= n => d <= (n-y)//3
        for d in range(1, (n - y) // 3 + 1):
            if (y + d) in S and (y + 2 * d) in S and (y + 3 * d) in S:
                return False
        
        # y = t1: t0, y, t2, t3. t0=y-d, t2=y+d, t3=y+2d.
        # Need y-d, y+d, y+2d in S.
        # Max d: y+2d <= n => d <= (n-y)//2. Also y-d >= 1 => d < y.
        max_d = min((n - y) // 2, y - 1)
        for d in range(1, max_d + 1):
            if (y - d) in S and (y + d) in S and (y + 2 * d) in S:
                return False
        
        # y = t2: t0, t1, y, t3. t0=y-2d, t1=y-d, t3=y+d.
        # Need y-2d, y-d, y+d in S.
        # Max d: y+d <= n => d <= n-y. Also y-2d >= 1 => 2d < y => d < y/2.
        max_d = min(n - y, (y - 1) // 2)
        for d in range(1, max_d + 1):
            if (y - 2 * d) in S and (y - d) in S and (y + d) in S:
                return False
        
        # y = t3: t0, t1, t2, y. t0=y-3d, t1=y-2d, t2=y-d.
        # Need y-3d, y-2d, y-d in S.
        # Max d: 3d < y => d < y/3.
        max_d = (y - 1) // 3
        for d in range(1, max_d + 1):
            if (y - 3 * d) in S and (y - 2 * d) in S and (y - d) in S:
                return False
                
        return True

    # Local search with restarts
    while time.time() < deadline:
        S = set(base3_set)
        candidates = list(set(range(1, n + 1)) - S)
        random.shuffle(candidates)
        
        # Greedy add
        for y in candidates:
            if can_add(y, S):
                S.add(y)
                
        # Try a second pass or just record
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Optional: Try to improve by removing and re-adding? 
        # For now, simple restarts are fast.
        
    return sorted(best_set)
