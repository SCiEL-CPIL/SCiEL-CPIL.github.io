import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    
    # weights[i] is 1/(i+1) for bit i
    weights = [1.0 / (i + 1) for i in range(N)]
    
    # conflict_mask[i] has bits set for all j that conflict with i
    # i, j correspond to numbers i+1, j+1
    conflict_mask = [0] * N
    
    # Precompute conflicts
    # Two numbers a, b (1-indexed) conflict if they are in a 3-AP.
    # 3-AP: a, m, c with m = (a+c)/2.
    # Or a, b, c with b = (a+c)/2.
    # For a given pair (x, y) with x < y (1-indexed):
    # They are in a 3-AP if:
    # 1. x, y are endpoints: (x+y) is even, mid = (x+y)//2 is integer and 1 <= mid <= N, mid != x, mid != y.
    # 2. x is middle: z = 2x - y. If 1 <= z <= N and z != x, z != y. (Since x<y, z<x)
    # 3. y is middle: z = 2y - x. If 1 <= z <= N and z != x, z != y. (Since x<y, z>y)
    
    # We iterate over all pairs (x, y) with 1 <= x < y <= N
    # If they conflict, set bits in conflict_mask[x-1] and conflict_mask[y-1]
    
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            is_conflict = False
            s = x + y
            if s % 2 == 0:
                m = s // 2
                if 1 <= m <= N and m != x and m != y:
                    is_conflict = True
            if not is_conflict:
                z = 2 * x - y
                if 1 <= z <= N and z != x and z != y:
                    is_conflict = True
            if not is_conflict:
                z = 2 * y - x
                if 1 <= z <= N and z != x and z != y:
                    is_conflict = True
            
            if is_conflict:
                conflict_mask[x - 1] |= (1 << (y - 1))
                conflict_mask[y - 1] |= (1 << (x - 1))

    # Precompute chunk sums for fast mask summation
    # Split 256 bits into 32 chunks of 8 bits
    CHUNKS = 32
    CHUNK_BITS = 8
    chunk_sums = [[0.0] * (1 << CHUNK_BITS) for _ in range(CHUNKS)]
    
    for c in range(CHUNKS):
        base = c * CHUNK_BITS
        for mask in range(1 << CHUNK_BITS):
            s = 0.0
            for b in range(CHUNK_BITS):
                if mask & (1 << b):
                    idx = base + b
                    if idx < N:
                        s += weights[idx]
            chunk_sums[c][mask] = s

    def mask_sum(mask):
        s = 0.0
        for c in range(CHUNKS):
            chunk_val = (mask >> (c * CHUNK_BITS)) & 0xFF
            s += chunk_sums[c][chunk_val]
        return s

    # Greedy initialization
    best_score = 0.0
    best_mask = 0
    current_mask = (1 << N) - 1
    current_score = 0.0
    temp_mask = current_mask
    while temp_mask:
        # Pick lowest set bit
        lsb = temp_mask & -temp_mask
        v = lsb.bit_length() - 1
        current_score += weights[v]
        current_mask |= lsb
        # Remove v and its conflicts from available
        temp_mask &= ~(lsb | conflict_mask[v])
    
    best_score = current_score
    best_mask = current_mask

    sys.setrecursionlimit(10000)
    
    def dfs(avail, score, current_set_mask):
        nonlocal best_score, best_mask
        
        if time.time() > deadline:
            return
            
        # Upper bound
        ub = score + mask_sum(avail)
        if ub <= best_score:
            return
            
        if avail == 0:
            if score > best_score:
                best_score = score
                best_mask = current_set_mask
            return
        
        # Pick vertex with highest weight (lowest index)
        lsb = avail & -avail
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New available: remove v and all its conflicts
        new_avail = avail & ~(lsb | conflict_mask[v])
        new_score = score + weights[v]
        new_set_mask = current_set_mask | lsb
        
        # Heuristic: If new_score + mask_sum(new_avail) > best_score, recurse
        # We let the recursive call handle the bound check to avoid double calculation
        dfs(new_avail, new_score, new_set_mask)
        
        # Branch 2: Exclude v
        # New available: remove v
        new_avail_ex = avail & ~lsb
        dfs(new_avail_ex, score, current_set_mask)

    initial_avail = (1 << N) - 1
    dfs(initial_avail, 0.0, 0)

    # Convert best_mask to list of integers (1-indexed)
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m &= ~lsb
        
    # Ensure non-empty (should be, greedy ensures it)
    if not result:
        result = [1]
        
    return result
