import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 85.0
    
    aps = []
    aps_by_element = [[] for _ in range(n+1)]
    
    for a in range(1, n+1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            ap = (a, b, c, e)
            idx = len(aps)
            aps.append(ap)
            for x in ap:
                aps_by_element[x].append(idx)
                
    in_S = [False] * (n+1)
    S_size = 0
    block_count = [0] * (n+1)
    S_list = []
    
    def add(x):
        nonlocal S_size
        if in_S[x]:
            return
        if block_count[x] != 0:
            return
        
        in_S[x] = True
        S_size += 1
        S_list.append(x)
        
        for ap_idx in aps_by_element[x]:
            ap = aps[ap_idx]
            for y in ap:
                if y == x:
                    continue
                others = [z for z in ap if z != x and z != y]
                if in_S[others[0]] and in_S[others[1]]:
                    block_count[y] += 1
                    
    def remove(x):
        nonlocal S_size
        if not in_S[x]:
            return
        
        in_S[x] = False
        S_size -= 1
        S_list.remove(x)
        
        for ap_idx in aps_by_element[x]:
            ap = aps[ap_idx]
            for y in ap:
                if y == x:
                    continue
                others = [z for z in ap if z != x and z != y]
                if in_S[others[0]] and in_S[others[1]]:
                    block_count[y] -= 1

    def fill():
        nonlocal S_size
        added_any = True
        while added_any:
            added_any = False
            for y in range(1, n+1):
                if not in_S[y] and block_count[y] == 0:
                    add(y)
                    added_any = True
                    break

    best_size = 0
    best_S = []
    
    while time.time() - start_time < time_limit:
        for i in range(1, n+1):
            in_S[i] = False
            block_count[i] = 0
        S_size = 0
        S_list = []
        
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        
        for x in candidates:
            if block_count[x] == 0:
                add(x)
        
        fill()
        
        if S_size > best_size:
            best_size = S_size
            best_S = list(S_list)
            
        kick_attempts = 0
        while kick_attempts < 100 and time.time() - start_time < time_limit:
            kick_attempts += 1
            if S_size == 0:
                break
            
            x = random.choice(S_list)
            
            in_S_backup = in_S.copy()
            block_count_backup = block_count.copy()
            S_size_backup = S_size
            S_list_backup = S_list.copy()
            
            remove(x)
            fill()
            
            if S_size > S_size_backup:
                if S_size > best_size:
                    best_size = S_size
                    best_S = list(S_list)
                kick_attempts = 0
            else:
                in_S = in_S_backup
                block_count = block_count_backup
                S_size = S_size_backup
                S_list = S_list_backup

    return best_S
