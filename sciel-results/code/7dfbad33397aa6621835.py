import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    
    # Precompute LCM of 1..12
    # LCM(1..12) = 27720
    period = 27720
    
    def has_matching(m, L):
        # For each i in 1..12, find positions j in [0, L-1] where (m+1+j) % i == 0
        # Use bitmask for efficiency
        adj_mask = [0] * 12
        for i in range(1, 13):
            # (m + 1 + j) % i == 0 => j ≡ (-m-1) mod i
            offset = (-m - 1) % i
            mask = 0
            for j in range(offset, L, i):
                mask |= (1 << j)
            adj_mask[i - 1] = mask
        
        # Kuhn's algorithm with bitmask adjacency
        match_to_right = [-1] * L
        
        def dfs(i, visited):
            mask = adj_mask[i]
            mm = mask
            while mm:
                lowbit = mm & (-mm)
                j = lowbit.bit_length() - 1
                mm ^= lowbit
                if visited[j]:
                    continue
                visited[j] = True
                if match_to_right[j] == -1 or dfs(match_to_right[j], visited):
                    match_to_right[j] = i
                    return True
            return False
        
        for i in range(12):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True

    best_L = 0
    best_m = 0
    
    # Iterate through one period
    for m in range(period):
        if time.time() > deadline:
            break
            
        # Find the smallest L such that has_matching(m, L) is True
        for L in range(12, 145):
            if has_matching(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                break
        
    return best_m
