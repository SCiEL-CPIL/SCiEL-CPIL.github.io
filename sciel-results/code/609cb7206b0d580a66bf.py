import random
import time

def construct():
    N = 256
    
    def check_add(x, S):
        # Check if adding x to S creates a 3-AP
        # x, a, b (x start)
        # a, x, b (x middle)
        # a, b, x (x end)
        
        # We can iterate d
        for d in range(1, N + 1):
            # x is middle: x-2d, x, x+2d
            a = x - 2 * d
            b = x + 2 * d
            if a >= 1 and b <= N:
                if a in S and b in S:
                    return False
            
            # x is start: x, x+d, x+2d
            b = x + d
            c = x + 2 * d
            if c <= N:
                if b in S and c in S:
                    return False
            
            # x is end: x-2d, x-d, x
            a = x - 2 * d
            b = x - d
            if a >= 1:
                if a in S and b in S:
                    return False
        return True

    # Initialize with greedy
    S = set()
    for x in range(1, N + 1):
        if check_add(x, S):
            S.add(x)
            
    current_score = sum(1.0 / x for x in S)
    best_score = current_score
    best_S = set(S)
    
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # Local Search
    while time.time() - start_time < time_limit:
        # Random move
        # 1. Try to add a random number
        # 2. Try to remove a random number
        # 3. Try to swap
        
        # To bias towards improvement, we can prioritize adding small numbers
        # But random exploration is needed to escape local minima.
        
        # Let's try adding a random number first, as it increases score.
        # If we can't add, we might need to remove something to make space.
        
        # Strategy:
        # 1. Pick a random candidate x from 1..N
        # 2. If x not in S:
        #    If check_add(x, S):
        #       Add x. Update score.
        #    Else:
        #       # x creates a conflict.
        #       # Identify the conflict?
        #       # Or just try to remove a conflicting element and then add x?
        #       # This is a "repair" move.
        #       # Let's find one element in S that conflicts with x.
        #       # If found, remove it, then add x.
        #       # This is a swap-like move.
        
        # Let's stick to simple Add/Remove/Swap for now, but make Add more frequent.
        
        r = random.random()
        if r < 0.6:
            # Add
            x = random.randint(1, N)
            if x not in S:
                if check_add(x, S):
                    S.add(x)
                    current_score += 1.0 / x
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
        elif r < 0.8:
            # Remove
            # Pick a random element to remove
            # To avoid list(S), pick random x from 1..N and check if in S
            # Retry a few times if not in S
            removed = False
            for _ in range(10):
                x = random.randint(1, N)
                if x in S:
                    S.remove(x)
                    current_score -= 1.0 / x
                    removed = True
                    break
        else:
            # Swap
            # Pick x in S, y not in S
            # Remove x, add y if valid
            swapped = False
            for _ in range(10):
                x = random.randint(1, N)
                if x in S:
                    y = random.randint(1, N)
                    if y not in S:
                        # Try swap
                        S.remove(x)
                        if check_add(y, S):
                            S.add(y)
                            delta = 1.0 / y - 1.0 / x
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_S = set(S)
                            swapped = True
                        else:
                            S.add(x)
                        break # Only try one swap per iteration?
                        
    return list(best_S)
