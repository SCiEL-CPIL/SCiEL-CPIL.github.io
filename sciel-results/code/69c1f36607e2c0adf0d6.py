import time
import random
from itertools import combinations

# Pre-compute all 4-subsets of [1, 16] as bitmasks
ALL_SETS = []
for combo in combinations(range(16), 4):
    mask = 0
    for i in combo:
        mask |= (1 << i)
    ALL_SETS.append(mask)

def can_add(S, F):
    n = len(F)
    for i in range(n):
        A = F[i]
        AS = A & S
        for j in range(i + 1, n):
            B = F[j]
            AB = A & B
            if AB == AS and AB == (B & S):
                return False
    return True

def greedy_random(seed):
    random.seed(seed)
    candidates = ALL_SETS[:]
    random.shuffle(candidates)
    
    F = []
    for S in candidates:
        if can_add(S, F):
            F.append(S)
            
    return F

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit
    
    best_F = []
    best_size = 0
    
    seed = 42
    while time.time() < deadline:
        F = greedy_random(seed)
        if len(F) > best_size:
            best_size = len(F)
            best_F = F[:]
        seed += 1
        
    result = []
    for mask in best_F:
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
