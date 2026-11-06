import numpy as np
import random
import time

def score_set_fast(S):
    present = np.zeros(513, dtype=bool)
    for x in S:
        present[x] = True
        
    good_d = 0
    for d in range(1, 512):
        if np.any(present[:512-2*d+1] & present[d:512-d+1] & present[2*d:]):
            good_d += 1
    return good_d

def construct():
    start_time = time.time()
    time_budget = 110
    
    # Initial State: Dense Core
    S = list(range(1, 33))
    current_score = score_set_fast(S)
    best_S = S[:]
    best_score = current_score
    
    tabu_list = []
    tabu_size = 50
    
    while time.time() - start_time < time_budget:
        best_neighbor = None
        best_neighbor_score = -1
        best_move = None
        
        # Generate 20 random candidates
        for _ in range(20):
            idx = random.randint(0, 31)
            old_val = S[idx]
            
            # Tabu check
            if (idx, old_val) in tabu_list:
                continue
                
            new_val = random.randint(1, 512)
            if new_val in S and new_val != old_val:
                continue
                
            new_S = S[:]
            new_S[idx] = new_val
            new_score = score_set_fast(new_S)
            
            if new_score > best_neighbor_score:
                best_neighbor_score = new_score
                best_neighbor = new_S
                best_move = (idx, old_val)
                
        if best_neighbor is None:
            # No valid moves, clear tabu
            tabu_list = []
            continue
            
        # Accept best neighbor (Tabu Search rule)
        S = best_neighbor
        current_score = best_neighbor_score
        
        # Update Tabu List
        if best_move:
            tabu_list.append(best_move)
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)
                
        if current_score > best_score:
            best_S = S[:]
            best_score = current_score
            
    return best_S
