import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 145
    
    if n == 1:
        return "1"
    
    # Generate permutations
    perms = [(1,)]
    for m in range(2, n + 1):
        new_perms = []
        for p in perms:
            for k in range(m):
                win = []
                for j in range(m):
                    idx = k + j
                    if idx < m - 1:
                        win.append(p[idx])
                    elif idx == m - 1:
                        win.append(m)
                    else:
                        win.append(p[idx - m])
                new_perms.append(tuple(win))
        perms = new_perms
        
    N = len(perms)
    
    # Precompute overlap matrix
    overlap = [bytearray(N) for _ in range(N)]
    for i in range(N):
        p_i = perms[i]
        for j in range(N):
            if i == j: continue
            p_j = perms[j]
            for k in range(n - 1, 0, -1):
                if p_i[-k:] == p_j[:k]:
                    overlap[i][j] = k
                    break
                    
    def get_ov(i, j):
        return overlap[i][j]
        
    order = list(range(N))
    
    def total_ov(ord_list):
        return sum(get_ov(ord_list[i], ord_list[i+1]) for i in range(len(ord_list)-1))
        
    current_ov = total_ov(order)
    best_order = order[:]
    best_ov = current_ov
    
    # Simulated Annealing
    T = 10.0
    cool_rate = 0.999
    
    while time.time() < deadline:
        if T < 0.1:
            break
            
        i = random.randint(0, N - 2)
        j = i + 1
        
        # Calculate delta for adjacent swap
        p_im1 = i - 1 if i > 0 else None
        p_jp1 = j + 1 if j < N - 1 else None
        
        old_ov = 0
        if p_im1 is not None: old_ov += get_ov(order[p_im1], order[i])
        old_ov += get_ov(order[i], order[j])
        if p_jp1 is not None: old_ov += get_ov(order[j], order[p_jp1])
        
        new_ov = 0
        if p_im1 is not None: new_ov += get_ov(order[p_im1], order[j])
        new_ov += get_ov(order[j], order[i])
        if p_jp1 is not None: new_ov += get_ov(order[i], order[p_jp1])
        
        delta = new_ov - old_ov
        
        if delta > 0 or random.random() < math.exp(delta / T):
            order[i], order[j] = order[j], order[i]
            current_ov += delta
            
            if current_ov > best_ov:
                best_ov = current_ov
                best_order = order[:]
                
        T *= cool_rate
        
    # Reconstruct from best_order
    word = list(perms[best_order[0]])
    for i in range(1, N):
        k = get_ov(best_order[i-1], best_order[i])
        word.extend(perms[best_order[i]][k:])
        
    return "".join(map(str, word))
