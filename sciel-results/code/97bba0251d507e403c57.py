import time
import sys
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability for all masks
    comp = [[0] * total_masks for _ in range(total_masks)]
    
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            if (i & ~j) == 0 or (j & ~i) == 0:
                comp[i][j] = 1
                comp[j][i] = 1

    def count_comparables(fam):
        cnt = 0
        L = len(fam)
        for i in range(L):
            for j in range(i + 1, L):
                cnt += comp[fam[i]][fam[j]]
        return cnt

    # Greedy initialization
    family = []
    cur = 0
    family.append(cur)
    for i in range(n):
        cur |= (1 << i)
        family.append(cur)
    
    used = set(family)
    
    for _ in range(m - len(family)):
        best_gain = -1
        best_mask = None
        
        for s in range(total_masks):
            if s in used:
                continue
            
            gain = 0
            for f in family:
                if comp[s][f]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        
        if best_mask is None:
            break
            
        family.append(best_mask)
        used.add(best_mask)

    # Local search with simulated annealing
    best_family = family[:]
    current_score = count_comparables(best_family)
    best_score = current_score
    best_family_copy = best_family[:]
    
    deadline = time.time() + 85
    
    # SA parameters
    temp = 10.0
    cooling = 0.9995
    min_temp = 0.01
    
    random.seed(12345)
    
    while time.time() < deadline:
        # Compute contribs
        contribs = [0] * m
        for i in range(m):
            c = 0
            fi = best_family[i]
            for j in range(m):
                if i != j:
                    c += comp[fi][best_family[j]]
            contribs[i] = c
        
        # Precompute sum_comps
        fam_set = set(best_family)
        sum_comps = {}
        for s in range(total_masks):
            if s in fam_set:
                continue
            sc = 0
            for f in best_family:
                sc += comp[s][f]
            sum_comps[s] = sc
        
        # Find best 1-swap
        best_delta = -100
        best_i = -1
        best_s = -1
        
        for i in range(m):
            fi = best_family[i]
            max_val = -1
            best_s_for_i = -1
            
            for s, sc in sum_comps.items():
                val = sc - comp[s][fi]
                if val > max_val:
                    max_val = val
                    best_s_for_i = s
            
            delta = max_val - contribs[i]
            if delta > best_delta:
                best_delta = delta
                best_i = i
                best_s = best_s_for_i
        
        # Accept or reject
        if best_delta >= 0:
            accept = True
        else:
            # Accept worse move with probability exp(delta/temp)
            if temp > min_temp:
                import math
                prob = math.exp(best_delta / temp)
                accept = random.random() < prob
            else:
                accept = False
        
        if accept and best_i != -1:
            best_family[best_i] = best_s
            current_score += best_delta
            
            if current_score > best_score:
                best_score = current_score
                best_family_copy = best_family[:]
        
        # Cool down
        temp *= cooling
        
        # If temperature is too low, reheat
        if temp < min_temp:
            temp = 5.0
    
    best_family = best_family_copy
    
    # Convert to list of lists
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result
