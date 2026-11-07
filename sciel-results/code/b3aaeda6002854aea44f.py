import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    S = set()
    # Try to build a large set
    # Heuristic: Start with large numbers, try to add smaller ones
    
    # Initial seed: Upper half
    # But let's do a greedy check from 128 down to 1
    
    for i in range(128, 0, -1):
        if time.time() > deadline:
            break
        
        # Check if i can be added
        # Conflict if there exist distinct a, b in S such that
        # i+a in S, i+b in S, a+b in S
        # Note: a,b must be distinct from i and each other.
        
        can_add = True
        # Optimize: only check pairs a,b that could form sums in S
        # Since S is a set, we can iterate through S
        # To speed up, we can break early
        
        # We need to check all pairs (a,b) in S
        # If S is large, this is O(|S|^2).
        # |S| max 128. 128^2 = 16384.
        # 128 * 16384 ~ 2M ops. Fast enough.
        
        S_list = list(S)
        n = len(S_list)
        
        for idx_a in range(n):
            a = S_list[idx_a]
            for idx_b in range(idx_a + 1, n):
                b = S_list[idx_b]
                
                # Check condition
                if (i + a in S) and (i + b in S) and (a + b in S):
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            S.add(i)
            
    # Convert to list of ints
    result = list(S)
    # Ensure types are correct
    return [int(x) for x in result]
