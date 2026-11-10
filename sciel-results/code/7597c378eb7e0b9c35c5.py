import numpy as np
import random
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    adj = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i + d) % m
            mask |= (1 << j)
        adj[i] = mask
    
    best_set = 0
    best_size = 0
    
    def calc_conflicts(S):
        c = 0
        temp = S
        while temp:
            lsb = temp & -temp
            i = lsb.bit_length() - 1
            c += (S & adj[i]).bit_count()
            temp ^= lsb
        return c // 2
    
    def greedy_init():
        S = 0
        nodes = list(range(m))
        random.shuffle(nodes)
        for n in nodes:
            if (S & adj[n]) == 0:
                S |= (1 << n)
        return S
    
    def local_improve(S, max_iters=50000):
        penalty = 20
        current_conf = calc_conflicts(S)
        current_size = S.bit_count()
        
        for _ in range(max_iters):
            idx = random.randint(0, m - 1)
            bit = 1 << idx
            
            if S & bit:
                neigh = (S & adj[idx]).bit_count()
                delta_conf = -neigh
                delta_size = -1
            else:
                neigh = (S & adj[idx]).bit_count()
                delta_conf = neigh
                delta_size = 1
                
            delta_energy = -delta_size + penalty * delta_conf
            
            if delta_energy < 0:
                S ^= bit
                current_conf += delta_conf
                current_size += delta_size
            elif delta_conf == 0 and delta_size == 1:
                S ^= bit
                current_conf += delta_conf
                current_size += delta_size
        
        return S
    
    def run_sa(start_state, iterations, t_start, t_end):
        nonlocal best_set, best_size
        
        S = start_state
        penalty = 20
        current_conf = calc_conflicts(S)
        current_size = S.bit_count()
        
        if current_conf == 0 and current_size > best_size:
            best_size = current_size
            best_set = S
        
        for i in range(iterations):
            t = t_start + (t_end - t_start) * (i / iterations)
            
            idx = random.randint(0, m - 1)
            bit = 1 << idx
            
            if S & bit:
                neigh = (S & adj[idx]).bit_count()
                delta_conf = -neigh
                delta_size = -1
            else:
                neigh = (S & adj[idx]).bit_count()
                delta_conf = neigh
                delta_size = 1
                
            delta_energy = -delta_size + penalty * delta_conf
            
            accept = False
            if delta_energy < 0:
                accept = True
            else:
                if t > 1e-9:
                    accept = random.random() < pow(2.718281828, -delta_energy / t)
            
            if accept:
                S ^= bit
                current_conf += delta_conf
                current_size += delta_size
                
                if current_conf == 0 and current_size > best_size:
                    best_size = current_size
                    best_set = S
    
    def final_polish(S):
        nonlocal best_set, best_size
        S = S
        for _ in range(5):
            S = local_improve(S, 20000)
            conf = calc_conflicts(S)
            size = S.bit_count()
            if conf == 0 and size > best_size:
                best_size = size
                best_set = S
            changed = True
            while changed:
                changed = False
                for n in range(m):
                    if not (S & (1 << n)) and (S & adj[n]) == 0:
                        S |= (1 << n)
                        changed = True
                        break
            if S.bit_count() > best_size and calc_conflicts(S) == 0:
                best_size = S.bit_count()
                best_set = S
        return S
    
    start_time = time.time()
    timeout = 550
    
    restart_count = 0
    while time.time() - start_time < timeout:
        restart_count += 1
        S = greedy_init()
        run_sa(S, 100000, 10.0, 0.1)
        S = final_polish(S)
    
    res = []
    for i in range(m):
        if best_set & (1 << i):
            res.append(i)
            
    return np.array(res, dtype=np.int64)
