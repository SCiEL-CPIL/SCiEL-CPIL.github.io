import time
import random
import math
import bisect

def _init_counts(S):
    counts = [0] * 256
    S_bits = 0
    for x in S:
        S_bits |= (1 << x)
    for d in range(1, 256):
        mask = S_bits & (S_bits >> d) & (S_bits >> (2 * d))
        counts[d] = bin(mask).count('1')
    return counts

def _get_affected_d(S, x):
    S_set = set(S)
    affected = set()
    for z in S_set:
        if z == x: continue
        diff = abs(x - z)
        affected.add(diff)
        if diff % 2 == 0:
            d = diff // 2
            if d > 0:
                affected.add(d)
    return affected

def _count_ap_involving(S_set, x, d):
    count = 0
    if (x - d) in S_set and (x + d) in S_set:
        count += 1
    if (x + d) in S_set and (x + 2*d) in S_set:
        count += 1
    if (x - 2*d) in S_set and (x - d) in S_set:
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAXV = 512
    
    best_score = -1
    best_list = []
    
    seeds = []
    seeds.append(list(range(1, 33)))
    seeds.append(list(range(1, 17)) + list(range(300, 316)))
    seeds.append(random.sample(range(1, MAXV + 1), N))
    
    for seed in seeds:
        if time.time() > deadline - 10: break
        
        current_list = seed[:]
        current_list.sort()
        S_set = set(current_list)
        
        counts = _init_counts(current_list)
        current_score = sum(1 for c in counts if c > 0)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        local_deadline = min(start_time + 10.0, deadline - 5.0)
        
        while time.time() < local_deadline:
            x_idx = random.randint(0, N - 1)
            x = current_list[x_idx]
            
            y = random.randint(1, MAXV)
            if y == x or y in S_set:
                continue
            
            D_x = _get_affected_d(current_list, x)
            S_temp = S_set - {x}
            D_y = _get_affected_d(list(S_temp), y)
            
            D_affected = D_x.union(D_y)
            
            delta = 0
            updates = {} # d: (rem, add)
            
            for d in D_affected:
                if d > 255: continue
                old_c = counts[d]
                rem = _count_ap_involving(S_set, x, d)
                add = _count_ap_involving(S_temp, y, d)
                new_c = old_c - rem + add
                
                if old_c == 0 and new_c > 0:
                    delta += 1
                elif old_c > 0 and new_c == 0:
                    delta -= 1
                
                updates[d] = (rem, add)
            
            new_score = current_score + delta
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                # Accept
                S_set.remove(x)
                S_set.add(y)
                
                # Update counts
                for d, (rem, add) in updates.items():
                    counts[d] -= rem
                    counts[d] += add
                
                # Update list
                current_list[x_idx] = y
                current_list.sort()
                
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
                
    return best_list
