import time

def construct():
    deadline = time.time() + 55.0
    
    S_by_k = [set() for _ in range(61)]
    S_by_k[0] = {0}
    current_set = []
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if len(current_set) >= 60:
            break
        
        n = len(current_set)
        
        new_S_by_k = [set() for _ in range(n + 2)]
        new_S_by_k[0] = set(S_by_k[0])
        
        for k in range(1, n + 1):
            new_S_by_k[k] = set(S_by_k[k])
            for s in S_by_k[k - 1]:
                new_S_by_k[k].add(s + x)
        
        valid = True
        for i in range(n + 1):
            if not valid:
                break
            if time.time() > deadline:
                valid = False
                break
            for j in range(i + 1, n + 1):
                if new_S_by_k[i] & new_S_by_k[j]:
                    valid = False
                    break
        
        if valid:
            S_by_k = new_S_by_k
            current_set.append(x)
    
    return list(current_set)
