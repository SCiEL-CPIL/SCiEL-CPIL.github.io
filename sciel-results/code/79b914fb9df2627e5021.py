import time
import random
from math import gcd

MAX_N = 128
lcm_table = [[0] * (MAX_N + 1) for _ in range(MAX_N + 1)]
for i in range(1, MAX_N + 1):
    for j in range(1, MAX_N + 1):
        if i == j:
            lcm_table[i][j] = i
        else:
            lcm_table[i][j] = (i // gcd(i, j)) * j

def is_valid_addition(current_set, x, lcm_table):
    n = len(current_set)
    for i in range(n):
        a = current_set[i]
        la_x = lcm_table[a][x]
        for j in range(i + 1, n):
            b = current_set[j]
            l_ab = lcm_table[a][b]
            lb_x = lcm_table[b][x]
            if l_ab == la_x == lb_x:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # 1. Deterministic Greedy (Ascending)
    current_set = []
    for x in range(1, MAX_N + 1):
        if is_valid_addition(current_set, x, lcm_table):
            current_set.append(x)
    best_set = current_set[:]
    best_size = len(current_set)
    
    # 2. Deterministic Greedy (Descending)
    current_set = []
    for x in range(MAX_N, 0, -1):
        if is_valid_addition(current_set, x, lcm_table):
            current_set.append(x)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]

    # 3. Randomized Greedy Restarts
    # Use a fixed seed for reproducibility if needed, but random is fine
    while time.time() < deadline:
        candidates = list(range(1, MAX_N + 1))
        random.shuffle(candidates)
        
        current_set = []
        for x in candidates:
            if is_valid_addition(current_set, x, lcm_table):
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we reach a high size, we might want to try local search
            # But adding more restarts is cheap.
            
    return best_set
