import random
import time

def construct():
    N = 512
    best_set = set()
    start_time = time.time()
    max_time = 55
    
    while time.time() - start_time < max_time:
        S = [False] * (N + 1)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            valid = True
            
            # Define the 4 checks as (max_d, lambda d: check)
            # To avoid lambda overhead, we can just write the loops but order them.
            
            m1 = (N - x) // 3
            m2 = min(x - 1, (N - x) // 2)
            m3 = min((x - 1) // 2, N - x)
            m4 = (x - 1) // 3
            
            # Create a list of (max_d, type) and sort
            # Types: 1, 2, 3, 4
            checks = [(m1, 1), (m2, 2), (m3, 3), (m4, 4)]
            checks.sort(key=lambda item: item[0])
            
            for max_d, check_type in checks:
                if max_d <= 0:
                    continue
                
                if check_type == 1:
                    # x, x+d, x+2d, x+3d
                    for d in range(1, max_d + 1):
                        if S[x+d] and S[x+2*d] and S[x+3*d]:
                            valid = False
                            break
                            
                elif check_type == 2:
                    # x-d, x, x+d, x+2d
                    for d in range(1, max_d + 1):
                        if S[x-d] and S[x+d] and S[x+2*d]:
                            valid = False
                            break
                            
                elif check_type == 3:
                    # x-2d, x-d, x, x+d
                    for d in range(1, max_d + 1):
                        if S[x-2*d] and S[x-d] and S[x+d]:
                            valid = False
                            break
                            
                elif check_type == 4:
                    # x-3d, x-2d, x-d, x
                    for d in range(1, max_d + 1):
                        if S[x-3*d] and S[x-2*d] and S[x-d]:
                            valid = False
                            break
                
                if not valid:
                    break
            
            if valid:
                S[x] = True
        
        # Convert list to set
        current_set = {i for i in range(1, N + 1) if S[i]}
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return sorted(best_set)
