import time
import random

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Precompute conflict masks
    # masks[x] is a list of integers. Each integer is a bitmask of 3 elements.
    # If S & mask == mask, then adding x creates a 4-AP.
    masks = [[] for _ in range(N)]
    
    for x in range(N):
        # Type 1: x is start (a). Others: x+d, x+2d, x+3d
        # d can go up to (N-1-x)//3
        max_d = (N - 1 - x) // 3
        for d in range(1, max_d + 1):
            m = (1 << (x+d)) | (1 << (x+2*d)) | (1 << (x+3*d))
            masks[x].append(m)
            
        # Type 2: x is 2nd (a+d). Others: x-d, x+d, x+2d
        # x-d >= 0 => d <= x
        # x+2d < N => d < (N-x)/2
        max_d = min(x, (N - 1 - x) // 2)
        for d in range(1, max_d + 1):
            m = (1 << (x-d)) | (1 << (x+d)) | (1 << (x+2*d))
            masks[x].append(m)
            
        # Type 3: x is 3rd (a+2d). Others: x-2d, x-d, x+d
        # x-2d >= 0 => d <= x/2
        # x+d < N => d < N-x
        max_d = min(x // 2, N - 1 - x)
        for d in range(1, max_d + 1):
            m = (1 << (x-2*d)) | (1 << (x-d)) | (1 << (x+d))
            masks[x].append(m)
            
        # Type 4: x is 4th (a+3d). Others: x-3d, x-2d, x-d
        # x-3d >= 0 => d <= x/3
        max_d = x // 3
        for d in range(1, max_d + 1):
            m = (1 << (x-3*d)) | (1 << (x-2*d)) | (1 << (x-d))
            masks[x].append(m)
            
    best_size = 0
    best_set = set()
    candidates = list(range(N))
    
    while time.time() < deadline:
        random.shuffle(candidates)
        S = 0
        size = 0
        remaining = N
        
        for x in candidates:
            remaining -= 1
            if size + remaining <= best_size:
                break
            
            # Check if x can be added
            blocked = False
            for m in masks[x]:
                if (S & m) == m:
                    blocked = True
                    break
            
            if not blocked:
                S |= (1 << x)
                size += 1
                
        if size > best_size:
            best_size = size
            best_set = {i+1 for i in range(N) if (S >> i) & 1}

    return sorted(best_set)
