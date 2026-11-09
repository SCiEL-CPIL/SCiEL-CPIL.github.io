import random
import time

def construct():
    n = 512
    deadline = time.time() + 85.0
    
    def has_4ap_with_x(S, x):
        # 1. x, x+d, x+2d, x+3d
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return True
        
        # 2. x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return True
        
        # 3. x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return True
        
        # 4. x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return True
                
        return False

    def greedy_init(order):
        S = set()
        for x in order:
            if not has_4ap_with_x(S, x):
                S.add(x)
        return S

    best_S = set()
    best_size = 0
    
    # 1. Multiple Greedy Inits
    # Run 100 random greedy inits
    for i in range(100):
        if time.time() > deadline - 15: break
        order = list(range(1, n+1))
        random.shuffle(order)
        S = greedy_init(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # 2. Local Search
    S = best_S
    
    while time.time() < deadline:
        current_size = len(S)
        added = False
        
        # Try to add a random element
        for _ in range(200):
            x = random.randint(1, n)
            if x in S:
                continue
            if not has_4ap_with_x(S, x):
                S.add(x)
                added = True
                break
        
        if added:
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
            continue
        
        # If no addition, try swap
        # Pick random u in S to remove
        if S:
            u = random.choice(list(S))
            S.remove(u)
            swapped = False
            for _ in range(200):
                x = random.randint(1, n)
                if x == u or x in S:
                    continue
                if not has_4ap_with_x(S, x):
                    S.add(x)
                    swapped = True
                    break
            
            if swapped:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
            else:
                S.add(u)
        else:
            break

    return sorted(best_S)
