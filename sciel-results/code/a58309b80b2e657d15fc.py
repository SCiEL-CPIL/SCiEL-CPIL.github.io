import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 1000
    S = []
    in_S = [False] * (N + 1)
    counts = {}
    
    def is_valid_add(x, current_S, current_counts):
        if current_counts.get(2 * x, 0) >= 2:
            return False
        for z in current_S:
            if z >= x:
                continue
            s = x + z
            if current_counts.get(s, 0) >= 2:
                return False
        return True

    def do_add(x):
        nonlocal S, in_S, counts
        S.append(x)
        in_S[x] = True
        counts[2 * x] = counts.get(2 * x, 0) + 1
        for z in S:
            if z == x:
                continue
            if z < x:
                s = x + z
                counts[s] = counts.get(s, 0) + 1

    def do_remove(x):
        nonlocal S, in_S, counts
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        if counts[2 * x] == 0:
            del counts[2 * x]
        for z in S:
            if z == x:
                continue
            if z < x:
                s = x + z
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]

    def is_valid_swap(x, y):
        # Check if y can be added to S \ {x}
        # Sums involving y: y+y and y+z for z in S\{x}, z < y
        
        # Check 2*y
        s = 2 * y
        c = counts.get(s, 0)
        # Did x contribute to 2*y?
        # x + w = 2y => w = 2y - x
        w = 2 * y - x
        if 1 <= w <= N and in_S[w] and w != x and w <= x:
            c -= 1
        if c >= 2:
            return False
            
        for z in S:
            if z == x:
                continue
            if z >= y:
                continue
            s = y + z
            c = counts.get(s, 0)
            # Did x contribute to s?
            # x + w = s => w = s - x
            w = s - x
            if 1 <= w <= N and in_S[w] and w != x and w <= x:
                c -= 1
            if c >= 2:
                return False
        return True

    # 1. Greedy Initialization
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if is_valid_add(x, S, counts):
            do_add(x)
            
    # 2. Local Search
    improved = True
    no_improve_count = 0
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any element
        for y in range(1, N + 1):
            if time.time() > deadline:
                return S
            if in_S[y]:
                continue
            if is_valid_add(y, S, counts):
                do_add(y)
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
            
        # If no add, try swaps
        # Randomize the order of x to explore different basins
        S_list = list(S)
        random.shuffle(S_list)
        
        for x in S_list:
            if time.time() > deadline:
                return S
            for y in range(1, N + 1):
                if time.time() > deadline:
                    return S
                if in_S[y]:
                    continue
                if is_valid_swap(x, y):
                    do_remove(x)
                    do_add(y)
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
                
        if not improved:
            no_improve_count += 1
            # If stuck for a while, we could restart, but let's just stop to save time for other strategies if implemented.
            # For now, break if no improvement in a full sweep.
            break

    return S
