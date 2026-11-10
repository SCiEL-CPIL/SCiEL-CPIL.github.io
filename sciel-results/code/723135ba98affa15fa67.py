import time

# Precompute divisibility masks
PERIOD = 27720
divs = [0] * PERIOD
for x in range(PERIOD):
    mask = 0
    for i in range(1, 13):
        if x % i == 0:
            mask |= (1 << (i - 1))
    divs[x] = mask

def has_matching(m, L):
    valid = [set() for _ in range(13)]
    
    for j in range(L):
        x_mod = (m + j) % PERIOD
        mask = divs[x_mod]
        for i in range(1, 13):
            if mask & (1 << (i - 1)):
                valid[i].add(j)
                
    for i in range(1, 13):
        if not valid[i]:
            return False
            
    indices = list(range(1, 13))
    indices.sort(key=lambda i: len(valid[i]))
    
    # Greedy check
    used = set()
    greedy_success = True
    for i in indices:
        found = False
        for j in valid[i]:
            if j not in used:
                used.add(j)
                found = True
                break
        if not found:
            greedy_success = False
            break
            
    if greedy_success:
        return True
        
    # DFS
    def dfs(k, used_mask):
        if k == 12:
            return True
        i = indices[k]
        for idx in valid[i]:
            if not (used_mask & (1 << idx)):
                if dfs(k + 1, used_mask | (1 << idx)):
                    return True
        return False
        
    return dfs(0, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_m = 0
    best_L = 0
    
    for m in range(PERIOD):
        if time.time() > deadline:
            break
            
        # Find minimal L for this m
        # L is at most 144, but let's go up to 200 to be safe
        for L in range(1, 201):
            if has_matching(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                break
        
    return best_m
