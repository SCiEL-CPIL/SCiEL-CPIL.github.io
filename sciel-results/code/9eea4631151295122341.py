import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Target n values to try, descending
    # 34 is the known bound.
    for n in [34, 33, 32, 31, 30, 25, 20, 15, 10, 9]:
        if time.time() > deadline:
            break
            
        try:
            res = solve_n(n, deadline - time.time())
            if res is not None:
                return res
        except Exception as e:
            print(f"Error for n={n}: {e}")
            continue
            
    # Fallback: The seed solution for n=9
    from itertools import combinations
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

def solve_n(n, time_budget):
    # Precompute triples
    triples = list(combinations(range(n), 3))
    M = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4 and K5 neighbor groups
    # For each triple i, find all K4s and K5s it belongs to
    # And store the indices of the OTHER triples in those cliques
    
    k4_groups = []
    k5_groups = []
    
    # To speed up precomputation, we can just iterate
    # But we need the structure for vectorized flip cost
    
    # Actually, let's just store the full cliques for initial cost calculation
    # and the neighbor groups for flip calculation
    
    # For flip calculation:
    # k4_groups[i] is an array of shape (num_k4s_containing_i, 3)
    # k5_groups[i] is an array of shape (num_k5s_containing_i, 9)
    
    k4_groups_list = [[] for _ in range(M)]
    k5_groups_list = [[] for _ in range(M)]
    
    # Iterate over all K4s
    for combo in combinations(range(n), 4):
        # The 4 triples in this K4
        t_indices = [triple_to_idx[t] for t in combinations(combo, 3)]
        # For each triple in this K4, add the other 3 to its group list
        for i in range(4):
            others = [t_indices[j] for j in range(4) if j != i]
            k4_groups_list[t_indices[i]].append(others)
            
    # Iterate over all K5s
    for combo in combinations(range(n), 5):
        # The 10 triples in this K5
        t_indices = [triple_to_idx[t] for t in combinations(combo, 3)]
        for i in range(10):
            others = [t_indices[j] for j in range(10) if j != i]
            k5_groups_list[t_indices[i]].append(others)
            
    # Convert to numpy arrays for fast access
    # We need ragged arrays or padded arrays.
    # Padded is easier for vectorization if we use masks, but 
    # the groups have fixed size for a given i?
    # No, the number of K4s containing a triple varies?
    # For a fixed n, every triple is in exactly (n-3) K4s.
    # And exactly C(n-3, 2) K5s.
    # So the size is constant!
    
    num_k4_per_triple = n - 3
    num_k5_per_triple = (n - 3) * (n - 4) // 2
    
    k4_groups = np.zeros((M, num_k4_per_triple, 3), dtype=np.int32)
    k5_groups = np.zeros((M, num_k5_per_triple, 9), dtype=np.int32)
    
    for i in range(M):
        if len(k4_groups_list[i]) == num_k4_per_triple:
            k4_groups[i] = np.array(k4_groups_list[i])
        else:
            # Should not happen
            pass
            
        if len(k5_groups_list[i]) == num_k5_per_triple:
            k5_groups[i] = np.array(k5_groups_list[i])
        else:
            pass
            
    # Initial State
    state = np.random.randint(0, 2, size=M)
    
    # Calculate initial cost
    # We need the full list of K4s and K5s to sum up violations
    # Let's generate them again or store them
    all_k4s = []
    for combo in combinations(range(n), 4):
        t_indices = np.array([triple_to_idx[t] for t in combinations(combo, 3)])
        all_k4s.append(t_indices)
    all_k4s = np.array(all_k4s) # Shape (NumK4, 4)
        
    all_k5s = []
    for combo in combinations(range(n), 5):
        t_indices = np.array([triple_to_idx[t] for t in combinations(combo, 3)])
        all_k5s.append(t_indices)
    all_k5s = np.array(all_k5s) # Shape (NumK5, 10)
    
    def calc_cost(s):
        red_k4 = np.all(s[all_k4s] == 0, axis=-1).sum()
        blue_k5 = np.all(s[all_k5s] == 1, axis=-1).sum()
        return red_k4 + blue_k5
        
    current_cost = calc_cost(state)
    best_cost = current_cost
    best_state = state.copy()
    
    if current_cost == 0:
        return format_result(n, state)
        
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    steps = 0
    
    # Precompute random indices? No, just use random.randint
    
    while time.time() < start_time + time_budget:
        # Pick random index
        i = random.randint(0, M - 1)
        
        # Calculate delta
        # state[i] is 0 (Red) or 1 (Blue)
        if state[i] == 0:
            # Flip to 1 (Blue)
            # Red K4s containing i: if other 3 are 0, they become not-all-red.
            # So R4 decreases by count of groups where all 3 are 0.
            r4_decrease = np.all(k4_groups[i] == 0, axis=-1).sum()
            
            # Blue K5s containing i: if other 9 are 1, they become all-blue.
            # So B5 increases by count of groups where all 9 are 1.
            b5_increase = np.all(k5_groups[i] == 1, axis=-1).sum()
            
            delta = -r4_decrease + b5_increase
        else:
            # Flip to 0 (Red)
            # Red K4s containing i: if other 3 are 0, they become all-red.
            # So R4 increases by count of groups where all 3 are 0.
            r4_increase = np.all(k4_groups[i] == 0, axis=-1).sum()
            
            # Blue K5s containing i: if other 9 are 1, they become not-all-blue.
            # So B5 decreases by count of groups where all 9 are 1.
            b5_decrease = np.all(k5_groups[i] == 1, axis=-1).sum()
            
            delta = r4_increase - b5_decrease
            
        new_cost = current_cost + delta
        
        # Acceptance
        if new_cost < current_cost or (T > min_T and random.random() < np.exp(-delta / T)):
            state[i] = 1 - state[i]
            current_cost = new_cost
            
            if current_cost == 0:
                return format_result(n, state)
                
            if current_cost < best_cost:
                best_cost = current_cost
                best_state = state.copy()
                
            T *= cooling_rate
        else:
            # Optionally, do nothing or random restart
            pass
            
        steps += 1
        if steps % 10000 == 0:
            # Check time
            pass
            
    # If we didn't find 0 cost, return the best valid one?
    # No, we must return a VALID one.
    # If best_cost > 0, this n failed.
    return None

def format_result(n, state):
    # state is numpy array of 0/1
    # Need to return list of colors in lexicographic order
    # The state array is already in lexicographic order of triples
    # because we generated triples in that order.
    colors = state.tolist()
    return {'n': n, 'colors': colors}
