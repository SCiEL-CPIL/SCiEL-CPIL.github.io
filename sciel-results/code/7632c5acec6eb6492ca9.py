import time
import random
from math import gcd

N = 128
gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        g = gcd(i, j)
        gcd_table[i][j] = g
        gcd_table[j][i] = g

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = [1]
    best_size = 1
    all_nums = list(range(1, N + 1))
    
    def can_add(selected, x, g_table):
        n_sel = len(selected)
        for i in range(n_sel):
            a = selected[i]
            g_ax = g_table[a][x]
            for j in range(i + 1, n_sel):
                b = selected[j]
                g_ab = g_table[a][b]
                if g_ab == g_ax:
                    if g_table[b][x] == g_ax:
                        return False
        return True

    while time.time() < deadline:
        candidates = all_nums[:]
        random.shuffle(candidates)
        
        current_set = []
        
        for x in candidates:
            if can_add(current_set, x, gcd_table):
                current_set.append(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        if best_size == N:
            break

    return best_set
