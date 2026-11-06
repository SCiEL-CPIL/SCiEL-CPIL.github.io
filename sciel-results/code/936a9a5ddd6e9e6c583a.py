import time

def _check_addition(values, sums, x):
    tails = [x]
    total = x
    for a in reversed(values):
        total += a
        tails.append(total)
    for t in tails:
        if t in sums:
            return False, None
    return True, set(tails)

def construct():
    deadline = time.time() + 85.0
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    def dfs(values, sums, last):
        nonlocal best_seq
        if time.time() > deadline:
            return
        
        if len(values) > len(best_seq):
            best_seq = list(values)
            
        # Pruning: even if we pick all remaining numbers, we can't beat best
        if len(values) + (256 - last) <= len(best_seq):
            return
            
        if last >= 256:
            return
            
        for x in range(last + 1, 257):
            if time.time() > deadline:
                return
            ok, new_sums = _check_addition(values, sums, x)
            if ok:
                values.append(x)
                dfs(values, new_sums, x)
                values.pop()
                
    dfs([], set(), 0)
    
    return best_seq
