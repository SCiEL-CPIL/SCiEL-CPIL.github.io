import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    def full_validate(S):
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            a = S_sorted[i]
            for j in range(i+1, k):
                b = S_sorted[j]
                for l in range(j, k):
                    c = S_sorted[l]
                    if (b + c) % a == 0:
                        return False
        return True

    def find_conflict(S):
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            a = S_sorted[i]
            for j in range(i+1, k):
                b = S_sorted[j]
                for l in range(j, k):
                    c = S_sorted[l]
                    if (b + c) % a == 0:
                        return (a, b, c)
        return None

    # Start with the known best interval
    best_set = set(range(171, 257))
    best_size = 86
    
    # Try to improve by local search
    # We will try to add small numbers and remove large numbers if it maintains size or increases it
    
    current_set = set(best_set)
    
    # Attempt to find a larger set
    # Strategy: Randomly pick a set of size 87, repair it.
    # If repair keeps size 87, we win.
    
    for trial in range(500):
        if time.time() > deadline - 5:
            break
            
        # Random initialization biased towards larger numbers
        # But we need to check if 87 is possible
        # Let's try to add 170 to the base and see if we can fix it by removing 1 element
        # Base: {171..256}
        # Add 170. Conflict 254+256.
        # If we remove 256, we have {170..255}. Size 86.
        # To get 87, we need to add 2 and remove 1, or add 1 and remove 0.
        # Add 170 and 169?
        # 170 conflicts with 254,256.
        # 169 conflicts with 252,255.
        # If we remove 256 and 255, we have {169,170,171..254}. Size 86.
        # It seems hard to gain size.
        
        # Let's try a general random repair
        S = set()
        # Pick 87 random numbers
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        S = set(candidates[:87])
        
        # Repair
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            conflict = find_conflict(S)
            if conflict:
                a, b, c = conflict
                # Remove one of them
                # Heuristic: remove the smallest
                to_remove = min(a, b, c)
                S.remove(to_remove)
                improved = True
        
        if len(S) > best_size and full_validate(S):
            best_set = set(S)
            best_size = len(S)
            if best_size >= 90:
                break
                
    # Also try greedy addition on the best set found
    if time.time() < deadline - 10:
        S = set(best_set)
        # Try adding any missing element
        for x in range(1, n+1):
            if time.time() > deadline - 5:
                break
            if x in S:
                continue
            S.add(x)
            if full_validate(S):
                best_set = set(S)
                best_size = len(S)
            else:
                S.remove(x)
                
    return list(best_set)
