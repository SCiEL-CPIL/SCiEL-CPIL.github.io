import time
import random

def construct():
    N = 512
    deadline = time.time() + 85.0 # Safety margin
    
    # Use a list for faster access than set
    # S_bool[i] is True if i is in S
    
    def can_add(x, S_bool):
        if S_bool[x]:
            return False
        
        # 1. x is a: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if S_bool[x+d] and S_bool[x+2*d] and S_bool[x+3*d]:
                return False
        
        # 2. x is a+d: x-d, x, x+d, x+2d
        # x-d >= 1 => d <= x-1
        # x+2d <= N => d <= (N-x)//2
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if S_bool[x-d] and S_bool[x+d] and S_bool[x+2*d]:
                return False
        
        # 3. x is a+2d: x-2d, x-d, x, x+d
        # x-2d >= 1 => d <= (x-1)//2
        # x+d <= N => d <= N-x
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if S_bool[x-2*d] and S_bool[x-d] and S_bool[x+d]:
                return False
        
        # 4. x is a+3d: x-3d, x-2d, x-d, x
        # x-3d >= 1 => d <= (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if S_bool[x-3*d] and S_bool[x-2*d] and S_bool[x-d]:
                return False
                
        return True

    best_set_bool = [False] * (N + 1)
    best_size = 0
    
    candidates = list(range(1, N + 1))
    
    # Phase 1: Random Greedy
    # Estimate: 1 pass ~ 0.2-0.5s. Allow ~50-100 passes.
    num_greedy_passes = 200
    
    for i in range(num_greedy_passes):
        if time.time() > deadline - 10.0:
            break
            
        random.shuffle(candidates)
        S_bool = [False] * (N + 1)
        current_size = 0
        
        for x in candidates:
            if can_add(x, S_bool):
                S_bool[x] = True
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_set_bool = S_bool.copy()
            
    # Phase 2: Local Search
    # Try to improve by removing one and refilling
    if time.time() < deadline - 5.0:
        current_best = best_set_bool.copy()
        current_size = best_size
        improved = True
        # Limit local search iterations to avoid timeout
        ls_iter = 0
        max_ls_iter = 50
        
        while improved and ls_iter < max_ls_iter and time.time() < deadline - 5.0:
            improved = False
            ls_iter += 1
            
            # Try removing a random element
            # Get list of indices where current_best is True
            elements = [i for i in range(1, N + 1) if current_best[i]]
            if not elements:
                break
            y = random.choice(elements)
            
            S_trial = current_best.copy()
            S_trial[y] = False
            size_trial = current_size - 1
            
            # Greedy fill
            # To speed up, we can just iterate 1..N
            for x in range(1, N + 1):
                if not S_trial[x]:
                    if can_add(x, S_trial):
                        S_trial[x] = True
                        size_trial += 1
                        
            if size_trial > current_size:
                current_best = S_trial
                current_size = size_trial
                best_set_bool = current_best
                best_size = current_size
                improved = True

    # Convert back to sorted list
    result = [i for i in range(1, N + 1) if best_set_bool[i]]
    return result
