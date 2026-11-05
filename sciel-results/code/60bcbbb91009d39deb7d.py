import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    fixed = [1, 2, 5, 11]
    
    # Max sum is 284 + 284 = 568
    # We need bits 0 to 568
    # Python ints are arbitrary precision, so this is fine.
    
    fixed_sums = 0
    for i, x in enumerate(fixed):
        for y in fixed[i:]:
            fixed_sums |= (1 << (x + y))
            
    candidates = []
    cand_fixed_sums = []
    
    for x in range(12, 285):
        s_fixed = (1 << (2*x)) | (1 << (x+1)) | (1 << (x+2)) | (1 << (x+5)) | (1 << (x+11))
        if not (s_fixed & fixed_sums):
            candidates.append(x)
            cand_fixed_sums.append(s_fixed)
            
    best_size = len(fixed)
    best_set = fixed[:]
    
    candidates.sort()
    pairs = list(zip(candidates, cand_fixed_sums))
    pairs.sort(key=lambda p: p[0])
    candidates = [p[0] for p in pairs]
    cand_fixed_sums = [p[1] for p in pairs]
    
    def dfs(selected, sums, idx):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
            
        current_size = len(selected) + len(fixed)
        if current_size > best_size:
            best_size = current_size
            best_set = fixed + selected
            
        if current_size + (len(candidates) - idx) <= best_size:
            return
            
        for i in range(idx, len(candidates)):
            if time.time() > deadline:
                return
                
            x = candidates[i]
            s_x_fixed = cand_fixed_sums[i]
            
            if s_x_fixed & sums:
                continue
                
            valid = True
            for s in selected:
                if (sums >> (x + s)) & 1:
                    valid = False
                    break
            if not valid:
                continue
                
            new_selected = selected + [x]
            
            new_sums = sums
            new_sums |= s_x_fixed
            for s in selected:
                new_sums |= (1 << (x + s))
                
            dfs(new_selected, new_sums, i + 1)
            
    dfs([], fixed_sums, 0)
    
    return best_set
