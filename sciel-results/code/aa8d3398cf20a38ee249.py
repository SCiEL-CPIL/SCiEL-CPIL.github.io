import time

def construct():
    n = 12
    start_time = time.time()
    deadline = start_time + 60.0  # 60s safety margin
    
    a = []
    sums = {0}
    
    for i in range(n):
        if time.time() > deadline:
            # Fallback to powers of 2
            return [1 << i for i in range(n)]
            
        candidate = 1
        while True:
            if time.time() > deadline:
                return [1 << i for i in range(n)]
                
            if candidate in a:
                candidate += 1
                continue
                
            # Check validity
            new_sums = set(sums)
            valid = True
            for s in sums:
                ns = s + candidate
                if ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
                
            if valid:
                a.append(candidate)
                sums = new_sums
                break
            candidate += 1
            
    return a
