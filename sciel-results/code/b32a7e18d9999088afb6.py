import numpy as np
import time
import random

def get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted([d for d in divs if d > 1])

def construct():
    N = 51975
    divisors = get_divisors(N)
    
    def greedy_build(order):
        classes = []
        arr = np.zeros(N, dtype=np.uint8)
        for m in order:
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = np.count_nonzero(arr[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            classes.append([m, best_r])
            arr[best_r::m] += 1
        return classes, arr, np.count_nonzero(arr == 0)
    
    # Initializations
    inits = []
    order_asc = sorted(divisors)
    c1, a1, s1 = greedy_build(order_asc)
    inits.append((c1, a1, s1))
    
    order_desc = sorted(divisors, reverse=True)
    c2, a2, s2 = greedy_build(order_desc)
    inits.append((c2, a2, s2))
    
    # Random init
    c3 = [[m, random.randint(0, m-1)] for m in divisors]
    a3 = np.zeros(N, dtype=np.uint8)
    for m, r in c3:
        a3[r::m] += 1
    s3 = np.count_nonzero(a3 == 0)
    inits.append((c3, a3, s3))
    
    best_classes = None
    best_arr = None
    best_score = float('inf')
    
    for c, a, s in inits:
        if s < best_score:
            best_score = s
            best_classes = [x[:] for x in c]
            best_arr = a.copy()
            
    # Local Search
    start_time = time.time()
    time_limit = 145.0
    
    indices = list(range(len(best_classes)))
    
    # To allow escaping local minima, we can do simulated annealing or just restarts.
    # Let's do restarts.
    # We will run local search for chunks of time.
    
    while time.time() - start_time < time_limit:
        # Run local search for this current state
        local_start = time.time()
        while time.time() - local_start < 5.0: # 5s chunks
            i = random.choice(indices)
            m, r_old = best_classes[i]
            r_new = random.randint(0, m - 1)
            if r_new == r_old:
                continue
                
            # Remove old
            slice_old = best_arr[r_old::m]
            become_uncov = np.count_nonzero(slice_old == 1)
            best_arr[r_old::m] -= 1
            mid_score = best_score + become_uncov
            
            # Add new
            slice_new = best_arr[r_new::m]
            become_cov = np.count_nonzero(slice_new == 0)
            best_arr[r_new::m] += 1
            new_score = mid_score - become_cov
            
            if new_score < best_score:
                best_score = new_score
                best_classes[i][1] = r_new
            else:
                # Reject
                best_arr[r_new::m] -= 1
                best_arr[r_old::m] += 1
                
        # If time remains, randomize and restart from best found so far?
        # Or just randomize completely?
        # Randomizing completely loses the structure.
        # Perturbing is better.
        # Let's perturb: change 10% of residues randomly.
        if time.time() - start_time < time_limit:
            for _ in range(len(best_classes) // 10):
                i = random.choice(indices)
                m = best_classes[i][0]
                best_classes[i][1] = random.randint(0, m - 1)
            
            # Recalculate arr and score
            best_arr = np.zeros(N, dtype=np.uint8)
            for m, r in best_classes:
                best_arr[r::m] += 1
            best_score = np.count_nonzero(best_arr == 0)
            
    return best_classes
