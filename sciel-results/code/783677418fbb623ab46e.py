import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 118.0
    
    N = 1000
    max_sum = 2 * N
    BIG = 10000
    
    def greedy_initial():
        in_set = [False] * (N + 1)
        counts = [0] * (max_sum + 1)
        for i in range(1, N + 1):
            if counts[2 * i] >= 2:
                continue
            ok = True
            for a in range(1, i):
                if in_set[a] and counts[a + i] >= 2:
                    ok = False
                    break
            if ok:
                in_set[i] = True
                counts[2 * i] += 1
                for a in range(1, i):
                    if in_set[a]:
                        counts[a + i] += 1
        return in_set, counts
    
    in_set, counts = greedy_initial()
    best_in_set = in_set[:]
    best_counts = counts[:]
    
    def compute_size_viol(in_set, counts):
        size = sum(1 for i in range(1, N + 1) if in_set[i])
        viol = 0
        for s in range(2, max_sum + 1):
            if counts[s] > 2:
                viol += counts[s] - 2
        return size, viol
    
    best_size, best_viol = compute_size_viol(best_in_set, best_counts)
    
    cur_in_set = in_set[:]
    cur_counts = counts[:]
    cur_size, cur_viol = compute_size_viol(cur_in_set, cur_counts)
    cur_energy = -cur_size + BIG * cur_viol
    
    T = 50.0
    T_min = 0.1
    cooling = 0.9998
    
    def apply_add(x, in_set, counts):
        in_set[x] = True
        counts[2 * x] += 1
        for a in range(1, x):
            if in_set[a]:
                counts[a + x] += 1
    
    def apply_remove(x, in_set, counts):
        in_set[x] = False
        counts[2 * x] -= 1
        for a in range(1, x):
            if in_set[a]:
                counts[a + x] -= 1
    
    step = 0
    while time.time() - start_time < deadline:
        x = random.randint(1, N)
        
        if cur_in_set[x]:
            viol_change = 0
            if cur_counts[2 * x] > 2:
                viol_change -= 1
            for a in range(1, x):
                if cur_in_set[a]:
                    s = a + x
                    if cur_counts[s] > 2:
                        viol_change -= 1
            size_change = -1
            delta_e = 1 + BIG * viol_change
            
            apply_remove(x, cur_in_set, cur_counts)
            cur_size += size_change
            cur_viol += viol_change
            cur_energy += delta_e
        else:
            viol_change = 0
            if cur_counts[2 * x] >= 2:
                viol_change += 1
            for a in range(1, x):
                if cur_in_set[a]:
                    s = a + x
                    if cur_counts[s] >= 2:
                        viol_change += 1
            size_change = 1
            delta_e = -1 + BIG * viol_change
            
            apply_add(x, cur_in_set, cur_counts)
            cur_size += size_change
            cur_viol += viol_change
            cur_energy += delta_e
        
        accepted = delta_e < 0 or (T > T_min and random.random() < math.exp(-delta_e / T))
        if not accepted:
            if cur_in_set[x]:
                apply_remove(x, cur_in_set, cur_counts)
                cur_size -= 1
                cur_viol -= viol_change
                cur_energy -= delta_e
            else:
                apply_add(x, cur_in_set, cur_counts)
                cur_size += 1
                cur_viol -= viol_change
                cur_energy -= delta_e
        
        if cur_viol == 0 and cur_size > best_size:
            best_size = cur_size
            best_in_set = cur_in_set[:]
            best_counts = cur_counts[:]
        
        step += 1
        if step % 500 == 0:
            T = max(T_min, T * cooling)
    
    return [i for i in range(1, N + 1) if best_in_set[i]]
