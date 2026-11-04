import math
import random
import time

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    start_time = time.time()
    time_limit = 110
    
    # Focused search space
    B_vals = [3, 4, 5]
    n_vals = [4, 5, 6, 7]
    K_vals = [15, 25, 35]
    restarts = 3
    max_steps = 10
    swaps_per_elem = 5
    
    for B in B_vals:
        for n in n_vals:
            if time.time() - start_time > time_limit:
                return best
                
            # Generate all values
            all_vals = []
            for mask in range(1 << n):
                val = 0
                m = mask
                j = 0
                while m:
                    if m & 1:
                        val += B ** j
                    m >>= 1
                    j += 1
                all_vals.append(val)
            
            N_full = len(all_vals)
            
            for K in K_vals:
                if K > N_full:
                    continue
                
                for _ in range(restarts):
                    if time.time() - start_time > time_limit:
                        return best
                        
                    # Initialize U
                    others = random.sample(range(1, N_full), K - 1)
                    U_set = {all_vals[i] for i in [0] + others}
                    current_score = calc_score(list(U_set))
                    
                    # Local Search
                    for step in range(max_steps):
                        if time.time() - start_time > time_limit:
                            return best
                            
                        improved = False
                        u_list = list(U_set)
                        random.shuffle(u_list)
                        
                        for u_elem in u_list:
                            for _ in range(swaps_per_elem):
                                v = random.choice(all_vals)
                                if v in U_set:
                                    continue
                                
                                new_U_set = U_set.copy()
                                new_U_set.remove(u_elem)
                                new_U_set.add(v)
                                
                                s = calc_score(list(new_U_set))
                                if s > current_score:
                                    current_score = s
                                    U_set = new_U_set
                                    improved = True
                                    break
                            if improved:
                                break
                        if not improved:
                            break
                    
                    if current_score > best_score:
                        best_score = current_score
                        best = list(U_set)

    return best
