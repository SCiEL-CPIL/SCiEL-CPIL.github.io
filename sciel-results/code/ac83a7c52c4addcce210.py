import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    evens = [2*k for k in range(1, 65)]
    n = len(evens)
    
    # Precompute gcd[a][b] for a,b in 1..64
    # gcd_val[i][j] = gcd(i+1, j+1)
    gcd_val = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_val[i][j] = gcd(i+1, j+1)
    
    def conflicts_involving(assignment, idx):
        """Count conflicts involving index idx."""
        count = 0
        r_idx = assignment[idx]
        for j in range(n):
            if j == idx:
                continue
            g = gcd_val[idx][j]
            if (r_idx - assignment[j]) % (2*g) == 0:
                count += 1
        return count
    
    def total_conflicts(assignment):
        total = 0
        for i in range(n):
            total += conflicts_involving(assignment, i)
        return total // 2  # each conflict counted twice
    
    # Initial: r_{2k} = k
    init = list(range(1, 65))
    tc = total_conflicts(init)
    
    best_solution = None
    if tc == 0:
        best_solution = init[:]
    
    if best_solution is None:
        current = init[:]
        cur_tc = tc
        best_local = current[:]
        best_tc = tc
        
        for step in range(200000):
            if time.time() > deadline:
                break
            if cur_tc == 0:
                best_solution = current[:]
                break
            
            i = random.randint(0, n-1)
            old_r = current[i]
            new_r = random.randint(0, evens[i]-1)
            if new_r == old_r:
                continue
            
            # Delta: remove old conflicts at i, add new
            old_conf_i = conflicts_involving(current, i)
            current[i] = new_r
            new_conf_i = conflicts_involving(current, i)
            delta = new_conf_i - old_conf_i
            
            if delta < 0 or random.random() < 0.1:  # accept improvements or 10% of others
                cur_tc += delta
                if cur_tc < best_tc:
                    best_tc = cur_tc
                    best_local = current[:]
            else:
                current[i] = old_r
        
        if best_tc == 0:
            best_solution = best_local
        elif best_solution is None:
            best_solution = best_local
    
    if best_solution is not None:
        return [[best_solution[i], evens[i]] for i in range(n)]
    
    # Fallback
    return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
