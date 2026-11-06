import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def solve(A, used_diffs, targets, M):
        if not targets:
            return A
        
        if len(A) > 32:
            return None
            
        if time.time() > deadline:
            return None
            
        last = A[-1]
        t_min = min(targets)
        
        max_x = min(M, last + t_min)
        
        for x in range(last + 1, max_x + 1):
            valid = True
            new_diffs = set()
            for a in A:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            new_targets = targets - new_diffs
            new_used = used_diffs | new_diffs
            
            res = solve(A + [x], new_used, new_targets, M)
            if res is not None:
                return res
                
        return None

    for M in range(30, 500):
        A = [0]
        used_diffs = set()
        targets = set(range(1, 31))
        
        res = solve(A, used_diffs, targets, M)
        if res is not None:
            return res
            
    return [0]
