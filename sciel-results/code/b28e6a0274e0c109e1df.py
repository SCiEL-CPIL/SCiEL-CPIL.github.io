import random
import time

def construct():
    n = 512
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    # Helper to check if adding x to S creates a 4-AP
    def is_safe(x, S):
        # S is a set
        # Check if x forms a 4-AP with any 3 elements in S
        # 4-AP: a, a+d, a+2d, a+3d
        # x can be a, a+d, a+2d, or a+3d
        
        # Case 1: x is the last term (a+3d)
        # a, a+d, a+2d in S
        # a = x - 3d
        # We need a, a+d, a+2d in S
        # Iterate d
        # Max d such that a >= 1 => x - 3d >= 1 => d <= (x-1)//3
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            if a in S and (a + d) in S and (a + 2 * d) in S:
                return False
        
        # Case 2: x is the third term (a+2d)
        # a, a+d, a+3d in S
        # a = x - 2d
        # a+3d = x + d
        # Need a, a+d, x+d in S
        # d can be such that a >= 1 => x - 2d >= 1 => d <= (x-1)//2
        # Also x+d <= n
        for d in range(1, min((x - 1) // 2, n - x) + 1):
            a = x - 2 * d
            if a in S and (a + d) in S and (x + d) in S:
                return False

        # Case 3: x is the second term (a+d)
        # a, a+2d, a+3d in S
        # a = x - d
        # a+2d = x + d
        # a+3d = x + 2d
        # Need x-d, x+d, x+2d in S
        # d such that x-d >= 1 => d <= x-1
        # x+2d <= n => d <= (n-x)//2
        for d in range(1, min(x - 1, (n - x) // 2) + 1):
            if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                return False

        # Case 4: x is the first term (a)
        # a+d, a+2d, a+3d in S
        # x+d, x+2d, x+3d in S
        # d such that x+3d <= n => d <= (n-x)//3
        for d in range(1, (n - x) // 3 + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return False

        return True

    # Generate base-3 set (digits 0,1)
    base3_set = set()
    for x in range(1, n + 1):
        y = x
        valid = True
        while y > 0:
            if y % 3 == 2:
                valid = False
                break
            y //= 3
        if valid:
            base3_set.add(x)
    
    best_set = base3_set.copy()
    
    candidates = [x for x in range(1, n + 1) if x not in base3_set]
    
    # Try multiple random restarts
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        random.shuffle(candidates)
        current_set = base3_set.copy()
        
        for x in candidates:
            if is_safe(x, current_set):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
        # If we have a very large set, we might stop early? 
        # Theoretical max is unknown, but > 150 is good.
        if len(best_set) > 200:
            break

    return sorted(best_set)
