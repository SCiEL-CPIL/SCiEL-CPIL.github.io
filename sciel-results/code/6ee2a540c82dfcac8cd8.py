import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def build_indices(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))
        
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))
            
        return num_triples, k4s, k5s

    def tabu_search(n, time_limit):
        num_triples, k4s, k5s = build_indices(n)
        
        # Convert to numpy arrays for vectorized cost calculation
        k4_arr = np.array(k4s, dtype=np.int32)
        k5_arr = np.array(k5s, dtype=np.int32)
        
        # Precompute reverse indices for potential future use (not strictly needed for vectorized cost)
        # But useful for "smart" moves.
        # For now, random moves are okay with Tabu.
        
        best_colors = None
        best_cost = float('inf')
        
        # Try a few different initializations
        inits = [
            lambda: np.random.randint(0, 2, size=num_triples, dtype=np.int8),
            lambda: np.zeros(num_triples, dtype=np.int8), # All red? Will have many red K4s.
            lambda: np.ones(num_triples, dtype=np.int8)   # All blue? Will have many blue K5s.
        ]
        
        # Actually, random is best.
        
        max_restarts = 20
        steps_per_restart = 5000
        
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
                
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
            # Initial cost
            red_k4 = np.sum((colors[k4_arr] == 0).sum(axis=1) == 4)
            blue_k5 = np.sum((colors[k5_arr] == 1).sum(axis=1) == 10)
            current_cost = red_k4 + blue_k5
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    return best_colors, True
            
            tabu_list = []
            tabu_tenure = 30
            no_improve_count = 0
            max_no_improve = 1000
            
            for i in range(steps_per_restart):
                if time.time() > time_limit:
                    break
                
                # Pick random triple
                t = np.random.randint(num_triples)
                
                # Tabu check
                if t in tabu_list:
                    continue
                    
                old_color = colors[t]
                new_color = 1 - old_color
                colors[t] = new_color
                
                # Update tabu
                tabu_list.append(t)
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)
                
                # Calculate new cost
                red_k4 = np.sum((colors[k4_arr] == 0).sum(axis=1) == 4)
                blue_k5 = np.sum((colors[k5_arr] == 1).sum(axis=1) == 10)
                new_cost = red_k4 + blue_k5
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors.copy()
                    no_improve_count = 0
                    if best_cost == 0:
                        return best_colors, True
                elif new_cost < current_cost:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                current_cost = new_cost
                
                if no_improve_count > max_no_improve:
                    # Kick: flip 10 random bits
                    for _ in range(10):
                        t_kick = np.random.randint(num_triples)
                        colors[t_kick] = 1 - colors[t_kick]
                    # Recalculate cost
                    red_k4 = np.sum((colors[k4_arr] == 0).sum(axis=1) == 4)
                    blue_k5 = np.sum((colors[k5_arr] == 1).sum(axis=1) == 10)
                    current_cost = red_k4 + blue_k5
                    no_improve_count = 0
                    tabu_list = [] # Clear tabu after kick
        
        return best_colors, (best_cost == 0)

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    targets = [34, 33, 32, 31, 30, 25, 20]
    
    time_alloc = {
        34: 50.0,
        33: 30.0,
        32: 20.0,
        31: 15.0,
        30: 10.0,
        25: 5.0,
        20: 2.0
    }
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        t_limit = time.time() + time_alloc.get(n, 5.0)
        
        try:
            colors, is_valid = tabu_search(n, t_limit)
            if is_valid:
                best_n = n
                best_colors = colors.tolist()
                if n >= 34:
                    break
        except Exception as e:
            pass

    return {'n': best_n, 'colors': best_colors}
