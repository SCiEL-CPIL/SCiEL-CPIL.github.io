import random
import time

def creates_ap(S, x, n=512):
    """
    Check if adding x to S creates a 4-term arithmetic progression.
    The AP must contain x.
    """
    # x is the 4th term: a, a+d, a+2d, x
    # a+3d = x => a = x-3d
    # d can range from 1 to x//3
    for d in range(1, x // 3 + 1):
        a = x - 3 * d
        if a >= 1:
            if a in S and (a + d) in S and (a + 2 * d) in S:
                return True

    # x is the 3rd term: a, a+d, x, a+3d
    # a+2d = x => a = x-2d
    # d can range from 1 to x//2
    for d in range(1, x // 2 + 1):
        a = x - 2 * d
        if a >= 1:
            b3 = a + 3 * d
            if b3 <= n:
                if a in S and (a + d) in S and b3 in S:
                    return True

    # x is the 2nd term: a, x, a+2d, a+3d
    # a+d = x => a = x-d
    # d can range from 1 to x-1
    for d in range(1, x):
        a = x - d
        if a >= 1:
            b2 = a + 2 * d
            b3 = a + 3 * d
            if b3 <= n:
                if a in S and b2 in S and b3 in S:
                    return True

    # x is the 1st term: x, x+d, x+2d, x+3d
    # d can range from 1 to (n-x)//3
    for d in range(1, (n - x) // 3 + 1):
        b1 = x + d
        b2 = x + 2 * d
        b3 = x + 3 * d
        if b3 <= n:
            if b1 in S and b2 in S and b3 in S:
                return True

    return False

def construct():
    n = 512
    best_set = set()
    best_size = 0
    
    start_time = time.time()
    max_time = 55.0
    
    # We can also try a deterministic greedy first to get a good baseline
    # Then randomize
    
    while time.time() - start_time < max_time:
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        current_set = set()
        
        for x in order:
            if not creates_ap(current_set, x, n):
                current_set.add(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    return sorted(best_set)
