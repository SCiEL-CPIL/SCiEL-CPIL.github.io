import time
import random
import itertools
from math import log
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    U_arr = np.array(U)
    # Vectorized computation of sums and differences
    sums = set()
    diffs = set()
    for i in range(n):
        sums.update(U_arr[i:] + U_arr[i])
        if i > 0:
            diffs.update((U_arr[i:] - U_arr[i]).tolist())
            diffs.update((U_arr[i] - U_arr[i:]).tolist())
    diffs.add(0)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def exhaustive_search(max_n, max_val, deadline):
    best_score = -1
    best_set = [0, 1, 3]
    
    # We include 0 in the set, so we choose n-1 elements from 1..max_val
    # Use a more efficient approach: try structured sets first
    for n in range(3, min(max_n, 6) + 1):
        # Try geometric-like sequences
        for base in range(2, max_val // (n - 1) + 1):
            if time.time() > deadline:
                return best_score, best_set
            # Generate base^k sequences
            seq = [base ** i for i in range(n - 1)]
            if max(seq) <= max_val:
                U = [0] + seq
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
        
        # Try arithmetic progressions
        for start in range(1, 10):
            for step in range(1, max_val // (n - 1) + 1):
                if time.time() > deadline:
                    return best_score, best_set
                seq = [start + i * step for i in range(n - 1)]
                if max(seq) <= max_val:
                    U = [0] + seq
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U[:]
        
        # Random sampling for larger n
        for _ in range(100):
            if time.time() > deadline:
                return best_score, best_set
            comb = random.sample(range(1, max_val + 1), n - 1)
            U = [0] + sorted(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    return best_score, best_set

def simulated_annealing(start_set, deadline):
    current = start_set[:]
    current_score = score_set(current)
    if current_score < 0:
        return -1, start_set
        
    best_score = current_score
    best_set = current[:]
    
    temp = 1.0
    cooling = 0.995
    min_temp = 0.005
    
    while time.time() < deadline and temp > min_temp:
        # Generate neighbor
        neighbor = current[:]
        move_type = random.random()
        
        if move_type < 0.6:
            # Shift an element with larger range
            i = random.randint(1, len(neighbor) - 1)
            delta = random.randint(-10, 10)
            if delta == 0: continue
            new_val = neighbor[i] + delta
            if new_val <= 0: continue
            neighbor[i] = new_val
            neighbor = sorted(set(neighbor))
            if len(neighbor) != len(current): continue
        elif move_type < 0.8:
            # Add an element
            if len(neighbor) < 15:
                new_val = random.randint(1, max(neighbor) + 20)
                if new_val not in neighbor:
                    neighbor.append(new_val)
                    neighbor.sort()
        elif move_type < 0.9:
            # Remove an element
            if len(neighbor) > 3:
                i = random.randint(1, len(neighbor) - 1)
                neighbor.pop(i)
        else:
            # Large jump: replace an element
            i = random.randint(1, len(neighbor) - 1)
            new_val = random.randint(1, max(neighbor) + 30)
            if new_val not in neighbor:
                neighbor[i] = new_val
                neighbor = sorted(set(neighbor))
                if len(neighbor) != len(current): continue
        
        new_score = score_set(neighbor)
        
        if new_score > current_score:
            current = neighbor
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current[:]
        else:
            # Accept with probability exp((new - current) / temp)
            if new_score < 0:
                pass
            else:
                delta_s = new_score - current_score
                if random.random() < np.exp(delta_s / temp):
                    current = neighbor
                    current_score = new_score
                    
        temp *= cooling
        
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s margin before 150s limit
    
    best_score = -1
    best_set = [0, 1, 3]
    
    # Phase 1: Structured search for good initial sets
    try:
        s1, set1 = exhaustive_search(6, 100, deadline - 30)
        if s1 > best_score:
            best_score = s1
            best_set = set1
    except Exception:
        pass
        
    # Phase 2: Simulated Annealing with multiple restarts
    restarts = 0
    while time.time() < deadline - 10 and restarts < 5:
        try:
            s2, set2 = simulated_annealing(best_set, deadline - 10)
            if s2 > best_score:
                best_score = s2
                best_set = set2
            restarts += 1
        except Exception:
            break
            
    return best_set

# EVOLVE-BLOCK-END
