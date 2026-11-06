import time

def construct():
    N = 4096
    max_sum = 3 * N
    used_sums = [False] * (max_sum + 1)
    S = []
    
    start_time = time.time()
    timeout = 85  # 5s margin
    
    for x in range(1, N + 1):
        if time.time() - start_time > timeout:
            break
            
        # Generate new sums involving x
        # 1. x + x + x
        # 2. x + x + s for s in S
        # 3. x + s_i + s_j for s_i, s_j in S (i <= j)
        
        new_sums = []
        
        # Optimization: If len(S) is large, this is slow.
        # But len(S) is expected to be small (< 50).
        
        new_sums.append(3 * x)
        
        for s in S:
            new_sums.append(2 * x + s)
            
        for i in range(len(S)):
            for j in range(i, len(S)):
                new_sums.append(x + S[i] + S[j])
        
        # Check for internal collisions
        # If there are duplicates in new_sums, reject
        if len(new_sums) != len(set(new_sums)):
            continue
            
        # Check for collisions with existing sums
        valid = True
        for s_val in new_sums:
            if used_sums[s_val]:
                valid = False
                break
                
        if valid:
            S.append(x)
            for s_val in new_sums:
                used_sums[s_val] = True
                
    return S
