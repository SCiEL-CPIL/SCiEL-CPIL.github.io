import random
import time

def construct():
    N = 512
    deadline = time.time() + 50.0  # 50s guard
    best_set = set()
    best_size = 0
    
    # Helper to check if x can be added to S
    def can_add(x, S):
        # Case 1: x is first term
        max_d1 = (N - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        # Case 2: x is second term
        max_d2 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        # Case 3: x is third term
        max_d3 = (x - 1) // 2
        for d in range(1, max_d3 + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        # Case 4: x is fourth term
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
        return True

    # Run many randomized greedy attempts
    while time.time() < deadline:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        current_set = set()
        
        for x in candidates:
            if can_add(x, current_set):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return sorted(best_set)
