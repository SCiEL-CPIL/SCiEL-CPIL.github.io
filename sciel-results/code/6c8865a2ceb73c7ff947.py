import time
import random
import math
from itertools import permutations
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    
    symbols = range(1, n)
    perms = list(permutations(symbols))
    N = len(perms) # 720 for n=7
    
    # Precompute overlap matrix
    overlap = np.zeros((N, N), dtype=np.int8)
    
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            max_k = n - 2
            found = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if pi[-k:] == pj[:k]:
                    found = k
                    break
            overlap[i][j] = found
            
    # Initial order: Lexicographic
    order = list(range(N))
    
    def get_score(order):
        s = 0
        for i in range(N - 1):
            s += overlap[order[i]][order[i+1]]
        return s
        
    best_score = get_score(order)
    best_order = order[:]
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    time_limit = deadline - 10.0
    
    while time.time() < time_limit:
        move_type = random.random()
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        
        if move_type < 0.5:
            # Swap
            old_score = get_score(order)
            order[i], order[j] = order[j], order[i]
            new_score = get_score(order)
            
            if new_score > old_score or random.random() < math.exp((new_score - old_score) / T):
                if new_score > best_score:
                    best_score = new_score
                    best_order = order[:]
            else:
                order[i], order[j] = order[j], order[i]
        else:
            # Reverse
            old_score = get_score(order)
            order[i:j+1] = reversed(order[i:j+1])
            new_score = get_score(order)
            
            if new_score > old_score or random.random() < math.exp((new_score - old_score) / T):
                if new_score > best_score:
                    best_score = new_score
                    best_order = order[:]
            else:
                order[i:j+1] = reversed(order[i:j+1])
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Final Polish: Adjacent Swap with Backtracking
    order = best_order[:]
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        i = 0
        while i < N - 1 and time.time() < deadline - 2.0:
            a = order[i]
            b = order[i+1]
            
            old_score = 0
            new_score = 0
            
            if i > 0:
                old_score += overlap[order[i-1]][a]
                new_score += overlap[order[i-1]][b]
            
            old_score += overlap[a][b]
            new_score += overlap[b][a]
            
            if i < N - 2:
                old_score += overlap[b][order[i+2]]
                new_score += overlap[a][order[i+2]]
            
            if new_score > old_score:
                order[i] = b
                order[i+1] = a
                improved = True
                if i > 0:
                    i -= 1
                else:
                    i += 1
            else:
                i += 1

    # Construct word
    word = []
    prev_idx = -1
    for curr_idx in best_order:
        p = perms[curr_idx]
        block = list(p) + [n] + list(p)
        
        if prev_idx == -1:
            word.extend(block)
        else:
            ov = overlap[prev_idx][curr_idx]
            word.extend(block[ov:])
        
        prev_idx = curr_idx
        
    return "".join(map(str, word))
